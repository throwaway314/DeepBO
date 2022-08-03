import sys
import logging
import warnings
import itertools
import subprocess
from tqdm.auto import tqdm, trange
import argparse
from pathlib import Path
import json
import numpy as np
import torch
import time
import pytorch_lightning as pl
import copy

# My imports
from weighted_retraining import GP_TRAIN_FILE, GP_OPT_FILE
from weighted_retraining.chem.chem_data import (
    WeightedJTNNDataset,
    WeightedMolTreeFolder,
)
from weighted_retraining.chem.jtnn.datautils import tensorize
from weighted_retraining.chem.chem_model import JTVAE
from weighted_retraining import utils
from weighted_retraining.chem.chem_utils import rdkit_quiet
from weighted_retraining.opt_scripts import base as wr_base

logger = logging.getLogger("chem-opt")


def setup_logger(logfile):
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)


def _run_command(command, command_name):
    logger.debug(f"{command_name} command:")
    logger.debug(command)
    start_time = time.time()
    run_result = subprocess.run(command, capture_output=True)
    # run_result = subprocess.run(command, stdout=sys.stdout, stderr=subprocess.STDOUT)
    assert run_result.returncode == 0, run_result.stderr
    logger.debug(f"{command_name} done in {time.time() - start_time:.1f}s")


def retrain_model(model, datamodule, save_dir, version_str, num_epochs, gpu):
    # Make sure logs don't get in the way of progress bars
    pl._logger.setLevel(logging.CRITICAL)
    train_pbar = utils.SubmissivePlProgressbar(process_position=1)

    # Create custom saver and logger
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=save_dir, version=version_str, name=""
    )
    checkpointer = pl.callbacks.ModelCheckpoint(save_last=True, monitor="loss/train", )

    # Handle fractional epochs
    if num_epochs < 1:
        max_epochs = 1
        limit_train_batches = num_epochs
    elif int(num_epochs) == num_epochs:
        max_epochs = int(num_epochs)
        limit_train_batches = 1.0
    else:
        raise ValueError(f"invalid num epochs {num_epochs}")

    # Create trainer
    trainer = pl.Trainer(
        gpus=1 if gpu else 0,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=1,
        checkpoint_callback=checkpointer,
        terminate_on_nan=True,
        logger=tb_logger,
        callbacks=[train_pbar],
        gradient_clip_val=20.0,  # Model is prone to large gradients
    )

    # Fit model
    trainer.fit(model, datamodule)


def _batch_decode_z_and_props(
        model: JTVAE,
        z: torch.Tensor,
        datamodule: WeightedJTNNDataset,
        args: argparse.Namespace,
        pbar: tqdm = None,
):
    """
    helper function to decode some latent vectors and calculate their properties
    """

    # Progress bar description
    if pbar is not None:
        old_desc = pbar.desc
        pbar.set_description("decoding")

    # Decode all points in a fixed decoding radius
    z_decode = []
    batch_size = 1
    for j in range(0, len(z), batch_size):
        with torch.no_grad():
            z_batch = z[j: j + batch_size]
            smiles_out = model.decode_deterministic(z_batch)
            if pbar is not None:
                pbar.update(z_batch.shape[0])
        z_decode += smiles_out

    # Now finding properties
    if pbar is not None:
        pbar.set_description("calc prop")

    # Find objective function values
    # Invalid points get a value of None
    z_prop = [
        args.invalid_score if s is None else
        datamodule.train_dataset.data_properties(datamodule.train_dataset.canonic_smiles.index(s))
        for s in z_decode
    ]

    # Now back to normal
    if pbar is not None:
        pbar.set_description(old_desc)

    return z_decode, z_prop


def _choose_best_rand_points(args: argparse.Namespace, dataset: WeightedMolTreeFolder):
    chosen_point_set = set()

    # Best scores at start
    targets_argsort = np.argsort(-dataset.data_properties.flatten())
    for i in range(args.n_best_points):
        chosen_point_set.add(targets_argsort[i])
    candidate_rand_points = np.random.choice(
        len(targets_argsort),
        size=args.n_rand_points + args.n_best_points,
        replace=False,
    )
    for i in candidate_rand_points:
        if i not in chosen_point_set and len(chosen_point_set) < (
                args.n_rand_points + args.n_best_points
        ):
            chosen_point_set.add(i)
    assert len(chosen_point_set) == (args.n_rand_points + args.n_best_points)
    chosen_points = sorted(list(chosen_point_set))

    return chosen_points


def _encode_mol_trees(model, mol_trees):
    batch_size = 64
    mu_list = []
    with torch.no_grad():
        for i in trange(
                0, len(mol_trees), batch_size, desc="encoding GP points", leave=False
        ):
            batch_slice = slice(i, i + batch_size)
            _, jtenc_holder, mpn_holder = tensorize(
                mol_trees[batch_slice], model.jtnn_vae.vocab, assm=False
            )
            tree_vecs, _, mol_vecs = model.jtnn_vae.encode(jtenc_holder, mpn_holder)
            muT = model.jtnn_vae.T_mean(tree_vecs)
            muG = model.jtnn_vae.G_mean(mol_vecs)
            mu = torch.cat([muT, muG], axis=-1).cpu().numpy()
            mu_list.append(mu)

    # Aggregate array
    mu = np.concatenate(mu_list, axis=0).astype(np.float32)
    return mu


def latent_optimization(
        args,
        model,
        datamodule,
        pooldatamodule,
        num_queries_to_do,
        gp_data_file,
        gp_run_folder,
        pbar=None,
        postfix=None,
):
    """ do latent space optimization with the optimal model (aka cheating) """

    ##################################################
    # Prepare GP
    ##################################################

    # First, choose GP points to train!
    dset = datamodule.train_dataset
    # chosen_indices = _choose_best_rand_points(args, dset)
    # mol_trees = [dset.data[i] for i in chosen_indices]
    # targets = dset.data_properties[chosen_indices]
    # chosen_smiles = [dset.canonic_smiles[i] for i in chosen_indices]
    mol_trees = dset.data
    targets = dset.data_properties
    chosen_smiles = dset.canonic_smiles

    # Next, encode these mol trees
    if args.gpu:
        model = model.cuda()
    latent_points = _encode_mol_trees(model, mol_trees)

    # Get the pool variables
    pooldset = pooldatamodule.train_dataset
    pool_mol_trees = pooldset.data
    pool_smiles = pooldset.canonic_smiles
    pool_latent_points = _encode_mol_trees(model, pool_mol_trees)
    pool_targets = pooldset.data_properties

    model = model.cpu()  # Make sure to free up GPU memory
    torch.cuda.empty_cache()  # Free the memory up for tensorflow

    starttime = time.time()

    # Save points to file
    def _save_gp_data(latent_points, targets, pool_latent_points, tottime=0):

        # Prevent overfitting to bad points
        targets = np.maximum(targets, args.invalid_score)
        targets = -targets.reshape(-1, 1)  # Since it is a minimization problem

        # Save the file
        np.savez_compressed(
            gp_data_file,
            X_train=latent_points.astype(np.float32),
            X_test=[],
            y_train=targets.astype(np.float32),
            y_test=[],
            smiles=chosen_smiles,
            pool_train=pool_latent_points.astype(np.float32),
            tottime=tottime
        )

    _save_gp_data(latent_points, targets, pool_latent_points)

    ##################################################
    # Run iterative GP fitting/optimization
    ##################################################
    curr_gp_file = None
    all_new_smiles = []
    all_new_props = []
    time_arr = []
    for gp_iter in range(num_queries_to_do):

        # print(gp_iter)

        # Part 1: fit GP
        # ===============================
        new_gp_file = gp_run_folder / f"gp_train_res{gp_iter:04d}.npz"
        log_path = gp_run_folder / f"gp_train{gp_iter:04d}.log"
        iter_seed = int(np.random.randint(10000))
        gp_train_command = [
            "python",
            GP_TRAIN_FILE,
            f"--nZ={args.n_inducing_points}",
            f"--seed={iter_seed}",
            f"--data_file={str(gp_data_file)}",
            f"--save_file={str(new_gp_file)}",
            f"--logfile={str(log_path)}",
        ]
        if gp_iter == 0:

            # Add commands for initial fitting
            gp_fit_desc = "GP initial fit"
            gp_train_command += [
                "--init",
                "--kmeans_init",
            ]
        else:
            gp_fit_desc = "GP incremental fit"
            gp_train_command += [
                f"--gp_file={str(curr_gp_file)}",
                f"--n_perf_measure=1",  # specifically see how well it fits the last point!
            ]

        # Set pbar status for user
        if pbar is not None:
            old_desc = pbar.desc
            pbar.set_description(gp_fit_desc)

        # Run command
        _run_command(gp_train_command, f"GP train {gp_iter}")
        curr_gp_file = new_gp_file

        # Part 2: optimize GP acquisition func to query point
        # ===============================

        # Run GP opt script
        opt_path = gp_run_folder / f"gp_opt_res{gp_iter:04d}.npy"
        log_path = gp_run_folder / f"gp_opt_{gp_iter:04d}.log"
        gp_opt_command = [
            "python",
            GP_OPT_FILE,
            f"--seed={iter_seed}",
            f"--gp_file={str(curr_gp_file)}",
            f"--data_file={str(gp_data_file)}",
            f"--save_file={str(opt_path)}",
            f"--n_out={1}",  # hard coded
            f"--logfile={str(log_path)}",
        ]
        if pbar is not None:
            pbar.set_description("optimizing acq func")
        _run_command(gp_opt_command, f"GP opt {gp_iter}")

        # Load point
        ind_opt = np.load(opt_path)
        z_opt = pool_latent_points[ind_opt]

        # # Decode point
        # smiles_opt, prop_opt = _batch_decode_z_and_props(
        #     model,
        #     torch.as_tensor(z_opt, device=model.device),
        #     pooldatamodule,
        #     args,
        #     pbar=pbar,
        # )

        # Get data for the chosen point
        smiles_opt = [pool_smiles[i] for i in ind_opt]
        prop_opt = pool_targets[ind_opt]
        # Delete chosen point from the pool
        for i in sorted(ind_opt, reverse=True):
            del pool_smiles[i]
            del pool_mol_trees[i]
        pool_targets = np.delete(pool_targets, ind_opt, axis=0)
        pool_latent_points = np.delete(pool_latent_points, ind_opt, axis=0)

        time_arr.append(time.time() - starttime)

        # Reset pbar description
        if pbar is not None:
            pbar.set_description(old_desc)

            # Update best point in progress bar
            if postfix is not None:
                postfix["best"] = max(postfix["best"], float(max(prop_opt)))
                pbar.set_postfix(postfix)

        # Append to new GP data
        latent_points = np.concatenate([latent_points, z_opt], axis=0)
        targets = np.concatenate([targets, prop_opt], axis=0)
        _save_gp_data(latent_points, targets, pool_latent_points, time_arr)

        # print((gp_iter, max(targets), len(latent_points)))

        # Append to overall list
        all_new_smiles += smiles_opt
        all_new_props += list(prop_opt)

    # Update datamodule with ALL data points
    return all_new_smiles, all_new_props


def latent_sampling(args, model, datamodule, num_queries_to_do, pbar=None):
    """ Draws samples from latent space and appends to the dataset """

    z_sample = torch.randn(num_queries_to_do, model.latent_dim, device=model.device)
    z_decode, z_prop = _batch_decode_z_and_props(
        model, z_sample, datamodule, args, pbar=pbar
    )

    return z_decode, z_prop


def main_loop(args):
    # Seeding
    pl.seed_everything(args.seed)

    # Make results directory
    result_dir = Path(args.result_root).resolve()
    result_dir.mkdir(parents=True)
    data_dir = result_dir / "data"
    data_dir.mkdir()
    setup_logger(result_dir / "log.txt")

    # Load data
    # trainargs = copy.deepcopy(args)
    # trainargs.property_file = None
    datamodule = WeightedJTNNDataset(args, utils.DataWeighter(args))
    datamodule.setup("fit")

    # Load data for the pool
    pooldatamodule = copy.deepcopy(datamodule)

    # print("hello")
    # print(datamodule.train_dataset.data_properties)
    # print(len(datamodule.train_dataset.data_properties))
    # print(len(datamodule.train_dataset.canonic_smiles))

    # Initial points
    init_idx = np.random.choice(range(len(pooldatamodule.train_dataset.data_properties)), size=5, replace=False)
    init_mol = [pooldatamodule.train_dataset.data[i] for i in init_idx]
    init_smiles = [pooldatamodule.train_dataset.canonic_smiles[i] for i in init_idx]
    init_prop = [pooldatamodule.train_dataset.data_properties[i] for i in init_idx]
    for i in sorted(init_idx, reverse=True):
        del pooldatamodule.train_dataset.canonic_smiles[i]
        del pooldatamodule.train_dataset.data[i]
    pooldatamodule.train_dataset.data_properties = np.delete(pooldatamodule.train_dataset.data_properties, init_idx)
    #
    # print(len(pooldatamodule.train_dataset.data_properties))
    # print(len(pooldatamodule.train_dataset.canonic_smiles))

    datamodule.train_dataset.data = init_mol
    datamodule.train_dataset.canonic_smiles = init_smiles
    datamodule.train_dataset.data_properties = np.array(init_prop)

    # Load model
    model = JTVAE.load_from_checkpoint(
        args.pretrained_model_file, vocab=datamodule.vocab
    )
    model.beta = model.hparams.beta_final  # Override any beta annealing

    # Set up some stuff for the progress bar
    num_retrain = int(np.ceil(args.query_budget / args.retraining_frequency))
    postfix = dict(
        retrain_left=num_retrain,
        best=-float("inf"),
        n_train=len(datamodule.train_dataset.data),
    )

    # Set up results tracking
    results = dict(
        opt_points=[],
        opt_point_properties=[],
        opt_model_version=[],
        params=str(sys.argv),
        sample_points=[],
        sample_versions=[],
        sample_properties=[],
    )

    # Main loop
    with tqdm(
            total=args.query_budget, dynamic_ncols=True, smoothing=0.0, file=sys.stdout
    ) as pbar:

        for ret_idx in range(num_retrain):
            pbar.set_postfix(postfix)
            pbar.set_description("retraining")

            # Decide whether to retrain
            samples_so_far = args.retraining_frequency * ret_idx

            # Update progress bar
            postfix["retrain_left"] -= 1
            pbar.set_postfix(postfix)

            # Do querying!
            pbar.set_description("querying")
            num_queries_to_do = min(
                args.retraining_frequency, args.query_budget - samples_so_far
            )
            if args.lso_strategy == "opt":
                gp_dir = result_dir / "gp" / f"iter{samples_so_far}"
                gp_dir.mkdir(parents=True)
                gp_data_file = gp_dir / "data.npz"
                x_new, y_new = latent_optimization(
                    args,
                    model,
                    datamodule,
                    pooldatamodule,
                    num_queries_to_do,
                    gp_data_file=gp_data_file,
                    gp_run_folder=gp_dir,
                    pbar=pbar,
                    postfix=postfix,
                )
            elif args.lso_strategy == "sample":
                x_new, y_new = latent_sampling(
                    args, model, datamodule, num_queries_to_do, pbar=pbar,
                )
            else:
                raise NotImplementedError(args.lso_strategy)

            print((x_new, y_new))

            # Update dataset
            datamodule.append_train_data(x_new, y_new)

            # Add new results
            results["opt_points"] += x_new
            results["opt_point_properties"] += y_new
            results["opt_model_version"] += [ret_idx] * len(x_new)

            # Save results
            np.savez_compressed(str(result_dir / "results.npz"), **results)

            # Keep a record of the dataset here
            new_data_file = (
                    data_dir / f"train_data_iter{samples_so_far + num_queries_to_do}.txt"
            )
            with open(new_data_file, "w") as f:
                f.write("\n".join(datamodule.train_dataset.canonic_smiles))

            postfix["best"] = max(postfix["best"], float(max(y_new)))
            postfix["n_train"] = len(datamodule.train_dataset.data)
            pbar.set_postfix(postfix)


if __name__ == "__main__":
    # Otherwise decoding fails completely
    rdkit_quiet()

    # arguments and argument checking
    parser = argparse.ArgumentParser()
    parser = WeightedJTNNDataset.add_model_specific_args(parser)
    parser = utils.DataWeighter.add_weight_args(parser)
    parser = wr_base.add_common_args(parser)
    parser = wr_base.add_gp_args(parser)

    # Pytorch lightning raises some annoying unhelpful warnings
    # in this script (because it is set up weirdly)
    # therefore we suppress warnings
    warnings.filterwarnings("ignore")

    # Parse args and run main code
    args = parser.parse_args()
    main_loop(args)
