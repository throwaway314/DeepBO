import argparse
import os
import pickle
import time
import pandas as pd
import tabulate

import bayesopt
import kernels
from bayesopt.generate_test_graphs import *
from benchmarks import NAS101Cifar10, NAS201
from kernels import *

parser = argparse.ArgumentParser(description='NAS-BOWL')
parser.add_argument('--dataset', default='nasbench101', help='The benchmark dataset to run the experiments. '
                                                             'options = ["nasbench101", "nasbench201"].')
parser.add_argument('--task', default=['cifar10-valid'],
                    nargs="+", help='the benchmark task *for nasbench201 only*.')
parser.add_argument("--use_12_epochs_result", action='store_true',
                    help='Whether to use the statistics at the end of the 12th epoch, instead of using the final '
                         'statistics *for nasbench201 only*')
parser.add_argument('--n_repeat', type=int, default=20, help='number of repeats of experiments')
parser.add_argument("--data_path", default='data/')
parser.add_argument('--n_init', type=int, default=30, help='number of initialising points')
parser.add_argument("--max_iters", type=int, default=17, help='number of maximum iterations')
parser.add_argument('--pool_size', type=int, default=100, help='number of candidates generated at each iteration')
parser.add_argument('--mutate_size', type=int, help='number of mutation candidates. By default, half of the pool_size '
                                                    'is generated from mutation.')
parser.add_argument('--pool_strategy', default='mutate', help='the pool generation strategy. Options: random,'
                                                              'mutate')
parser.add_argument('--save_path', default='results/', help='path to save log file')
parser.add_argument('-s', '--strategy', default='gbo', help='optimisation strategy: option: gbo (graph bo), '
                                                            'random (random search)')
parser.add_argument('-a', "--acquisition", default='EI', help='the acquisition function for the BO algorithm. option: '
                                                              'UCB, EI, AEI')
parser.add_argument('-k', '--kernels', default=['wl'],
                    nargs="+",
                    help='graph kernel to use. This can take multiple input arguments, and '
                         'the weights between the kernels will be automatically determined'
                         ' during optimisation (weights will be deemed as additional '
                         'hyper-parameters.')
parser.add_argument('-p', '--plot', action='store_true', help='whether to plot the procedure each iteration.')
parser.add_argument('--batch_size', type=int, default=5, help='Number of samples to evaluate')
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--cuda', action='store_true', help='Whether to use GPU acceleration')
parser.add_argument('--fixed_query_seed', type=int, default=None,
                    help='Whether to use deterministic objective function as NAS-Bench-101 has 3 different seeds for '
                         'validation and test accuracies. Options in [None, 0, 1, 2]. If None the query will be '
                         'random.')
parser.add_argument('--load_from_cache', action='store_true', help='Whether to load the pickle of the dataset. ')
parser.add_argument('--mutate_unpruned_archs', action='store_true',
                    help='Whether to mutate on the unpruned archs. This option is only valid if mutate '
                         'is specified as the pool_strategy')
parser.add_argument('--no_isomorphism', action='store_true', help='Whether to allow mutation to return'
                                                                  'isomorphic architectures')
parser.add_argument('--maximum_noise', default=0.01, type=float, help='The maximum amount of GP jitter noise variance')
args = parser.parse_args()
options = vars(args)
print(options)

if args.seed is not None:
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

if args.cuda and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

assert args.strategy in ['random', 'gbo']
assert args.pool_strategy in ['random', 'mutate', ]

# Initialise the objective function. Negative ensures a maximisation task that is assumed by the acquisition function.

# Persistent data structure...
cache_path = 'data/' + args.dataset + '.pickle'

o = None
if args.load_from_cache:
    if os.path.exists(cache_path):
        try:
            o = pickle.load(open(cache_path, 'rb'))
            o.seed = args.fixed_query_seed
            if args.dataset == 'nasbench201':
                o.task = args.task[0]
                o.use_12_epochs_result = args.use_12_epochs_result
        except:
            pass

if o is None:
    if args.dataset == 'nasbench101':
        o = NAS101Cifar10(data_dir=args.data_path, negative=True, seed=args.fixed_query_seed)
    elif args.dataset == 'nasbench201':
        o = NAS201(data_dir=args.data_path, negative=True, seed=args.fixed_query_seed, task=args.task[0],
                   use_12_epochs_result=args.use_12_epochs_result)

    else:
        raise NotImplementedError("Required dataset " + args.dataset + " is not implemented!")

all_data = []
for j in range(args.n_repeat):
    start_time = time.time()
    best_tests = []
    best_vals = []
    # 2. Take n_init_point random samples from the candidate points to warm_start the Bayesian Optimisation
    x, x_config, x_unpruned = random_sampling(args.n_init, benchmark=args.dataset, save_config=True,
                                              return_unpruned_archs=True)
    y_np_list = [o.eval(x_) for x_ in x]
    y = torch.tensor([y[0] for y in y_np_list]).float()
    train_details = [y[1] for y in y_np_list]

    print(x[0].nodes(data=True))
    print(x[0].edges(data=True))
    quit()

    # The test accuracy from NASBench. This is retrieved only for reference, and is not used in BO at all
    test = torch.tensor([o.test(x_) for x_ in x])
    # Initialise the GP surrogate and the acquisition function
    pool = x[:]
    unpruned_pool = x_unpruned[:]
    kern = []

    print(x)
    print(np.shape(x))
    print(y)
    print(np.shape(y))
    quit()

    for k in args.kernels:
        # Graph kernels
        if k == 'wl':
            k = WeisfilerLehman(h=2, oa=args.dataset != 'nasbench201',)
        elif k == 'mlk':
            k = MultiscaleLaplacian(n=1)
        elif k == 'vh':
            k = WeisfilerLehman(h=0, oa=args.dataset != 'nasbench201',)
        else:
            try:
                k = getattr(kernels, k)
                k = k()
            except AttributeError:
                logging.warning('Kernel type ' + str(k) + ' is not understood. Skipped.')
                continue
        kern.append(k)
    if kern is None:
        raise ValueError("None of the kernels entered is valid. Quitting.")
    if args.strategy != 'random':
        gp = bayesopt.GraphGP(x, y, kern, verbose=args.verbose)
        gp.fit(wl_subtree_candidates=(0,) if args.kernels[0] == 'vh' else tuple(range(1, 4)),
               optimize_lik=args.fixed_query_seed is None,
               max_lik=args.maximum_noise)
    else:
        gp = None

    # 3. Main optimisation loop
    columns = ['Iteration', 'Last func val', 'Best func val', 'True optimum in pool',
               'Pool regret', 'Last func test', 'Best func test', 'Time', 'TrainTime']

    res = pd.DataFrame(np.nan, index=range(args.max_iters), columns=columns)
    sampled_idx = []
    for i in range(args.max_iters):
        # Generate a pool of candidates from a pre-specified strategy
        if args.pool_strategy == 'random':
            pool, _, unpruned_pool = random_sampling(args.pool_size, benchmark=args.dataset, return_unpruned_archs=True)
        elif args.pool_strategy == 'mutate':
            pool, unpruned_pool = mutation(x, y, benchmark=args.dataset, pool_size=args.pool_size,
                                           n_best=10,
                                           n_mutate=args.mutate_size if args.mutate_size else args.pool_size // 2,
                                           observed_archs_unpruned=x_unpruned if args.mutate_unpruned_archs else None,
                                           allow_isomorphism=not args.no_isomorphism)
        else:
            pass

        if args.strategy != 'random':
            if args.acquisition == 'UCB':
                a = bayesopt.GraphUpperConfidentBound(gp)
            elif args.acquisition == 'EI':
                a = bayesopt.GraphExpectedImprovement(gp, in_fill='best', augmented_ei=False)
            elif args.acquisition == 'AEI':
                # Uses the augmented EI heuristic and changed the in-fill criterion to the best test location with
                # the highest *posterior mean*, which are preferred when the optimisation is noisy.
                a = bayesopt.GraphExpectedImprovement(gp, in_fill='posterior', augmented_ei=True)
            else:
                raise ValueError("Acquisition function" + str(args.acquisition) + ' is not understood!')
        else:
            a = None

        # Ask for a location proposal from the acquisition function
        if args.strategy == 'random':
            next_x = random.sample(pool, args.batch_size)
            sampled_idx.append(next_x)
            next_x_unpruned = None
        else:
            next_x, eis, indices = a.propose_location(top_n=args.batch_size, candidates=pool)
            next_x_unpruned = [unpruned_pool[i] for i in indices]
        # Evaluate this location from the objective function
        detail = [o.eval(x_) for x_ in next_x]
        next_y = [y[0] for y in detail]
        train_details += [y[1] for y in detail]
        next_test = [o.test(x_).item() for x_ in next_x]
        # Evaluate all candidates in the pool to obtain the regret (i.e. true best *in the pool* compared to the one
        # returned by the Bayesian optimiser proposal)
        pool_vals = [o.eval(x_)[0] for x_ in pool]
        if gp is not None:
            pool_preds = gp.predict(pool,)
            pool_preds = [p.detach().cpu().numpy() for p in pool_preds]
            pool.extend(next_x)

        # Update the GP Surrogate
        x.extend(next_x)
        if args.pool_strategy in ['mutate']:
            x_unpruned.extend(next_x_unpruned)
        y = torch.cat((y, torch.tensor(next_y).view(-1))).float()
        test = torch.cat((test, torch.tensor(next_test).view(-1))).float()

        if args.strategy != 'random':
            gp.reset_XY(x, y)
            gp.fit(wl_subtree_candidates=(0,) if args.kernels[0] == 'vh' else tuple(range(1, 4)),
                   optimize_lik=args.fixed_query_seed is None,
                   max_lik=args.maximum_noise
                   )

            # Compute the GP posterior distribution on the trainning inputs
            train_preds = gp.predict(x,)
            train_preds = [t.detach().cpu().numpy() for t in train_preds]

        zipped_ranked = list(sorted(zip(pool_vals, pool), key=lambda x: x[0]))[::-1]
        true_best_pool = np.exp(-zipped_ranked[0][0])

        # Updating evaluation metrics
        best_val = torch.exp(-torch.max(y))
        pool_regret = np.abs(np.exp(-np.max(next_y)) - true_best_pool)
        best_test = torch.exp(-torch.max(test))

        end_time = time.time()
        # Compute the cumulative training time.
        try:
            cum_train_time = np.sum([item['train_time'] for item in train_details]).item()
        except TypeError:
            cum_train_time = np.nan
        values = [str(i), str(np.exp(-np.max(next_y))), best_val.item(), true_best_pool, pool_regret,
                  str(np.exp(-np.max(next_test))), best_test.item(), str(end_time - start_time),
                  str(cum_train_time)]
        table = tabulate.tabulate([values], headers=columns, tablefmt='simple', floatfmt='8.4f')
        best_vals.append(best_val)
        best_tests.append(best_test)

        if i % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)

        if args.plot and args.strategy != 'random':
            import matplotlib.pyplot as plt

            plt.subplot(221)
            # True validation error vs GP posterior
            plt.title('Val')
            plt.plot(pool_vals, pool_vals, '.')
            plt.errorbar(pool_vals, pool_preds[0],
                         fmt='.', yerr=np.sqrt(np.diag(pool_preds[1])),
                         capsize=2, color='b', alpha=0.2)
            plt.grid(True)
            plt.subplot(222)
            # Acquisition function
            plt.title('Acquisition')
            plt.plot(pool_vals, eis, 'b+')
            plt.xlim([2.5, None])
            plt.subplot(223)
            plt.title('Train')

            y1, y2 = y[:-args.batch_size], y[-args.batch_size:]
            plt.plot(y, y, ".")
            plt.plot(y1, train_preds[0][:-args.batch_size], 'b+')
            plt.plot(y2, train_preds[0][-args.batch_size:], 'r+')

            if args.verbose:
                from perf_metrics import *
                print('Spearman: ', spearman(pool_vals, pool_preds[0]))
            plt.subplot(224)
            # Best metrics so far
            xaxis = np.arange(len(best_tests))
            plt.plot(xaxis, best_tests, "-.", c='C1', label='Best test so far')
            plt.plot(xaxis, best_vals, "-.", c='C2', label='Best validation so far')
            plt.legend()
            plt.show()

        res.iloc[i, :] = values
    all_data.append(res)

if args.save_path is not None:
    import datetime
    time_string = datetime.datetime.now()
    time_string = time_string.strftime('%Y%m%d_%H%M%S')
    args.save_path = os.path.join(args.save_path, time_string)
    if not os.path.exists(args.save_path): os.makedirs(args.save_path)
    pickle.dump(all_data, open(args.save_path + '/data.pickle', 'wb'))
    option_file = open(args.save_path + "/command.txt", "w+")
    option_file.write(str(options))
    option_file.close()
