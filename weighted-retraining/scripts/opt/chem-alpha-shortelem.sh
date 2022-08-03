#!/bin/bash

#SBATCH --gres=gpu:volta:1
#SBATCH -c 20
#SBATCH -o logs/chem-weighted-mingapalpha-%j-%a.out
#SBATCH -a 1-5

module load anaconda/2020b
# conda init
source activate weighted-retraining

# Meta flags
gpu="--gpu"  # change to "" if no GPU is to be used
seed_array=( 1 2 3 4 5 )
root_dir="logs/opt/chem-mingap"
start_model="logs/train/chem/lightning_logs/shortelem/checkpoints/last.ckpt"
query_budget=500
n_retrain_epochs=0.1
n_init_retrain_epochs=1

# Experiment 1: weighted retraining with different parameters
# ==================================================
k_high="1e-1"
k_low="1e-3"
r_high=100
r_low=50
r_inf="1000000"  # Set to essentially be infinite (since "inf" is not supported)
weight_type="rank"
lso_strategy="opt"

# Set specific experiments to do:
# normal LSO, our setting, retrain only, weight only, high weight/low retrain, high retrain/low weight
k_expt=(  "inf"    "$k_low" "inf"    "$k_low" "$k_low"  "$k_high" )
r_expt=(  "$r_inf" "$r_low" "$r_low" "$r_inf" "$r_high" "$r_low" )

#k_expt=(  "$k_low")
#r_expt=(  "$r_low")

k_expt=(  "inf")
r_expt=(  "$r_inf")

expt_index=0  # Track experiments
for seed in "${seed_array[@]}"; do
    for ((i=0;i<${#k_expt[@]};++i)); do

        # Increment experiment index
        expt_index=$((expt_index+1))

        # Break loop if using slurm and it's not the right task
        if [[ -n "${SLURM_ARRAY_TASK_ID}" ]] && [[ "${SLURM_ARRAY_TASK_ID}" != "$expt_index" ]]
        then
            continue
        fi


        # Echo info of task to be executed
        r="${r_expt[$i]}"
        k="${k_expt[$i]}"
        echo "r=${r} k=${k} seed=${seed}"

        # Run command
        python weighted_retraining/opt_scripts/opt_qm9.py \
            --seed="$seed" $gpu \
            --query_budget="$query_budget" \
            --retraining_frequency="$r" \
            --result_root="${root_dir}/seed${seed}" \
            --pretrained_model_file="$start_model" \
            --lso_strategy="$lso_strategy" \
            --train_path=data/chem/qm9/tensors_train \
            --vocab_file=data/chem/qm9/vocab.txt \
            --property_file=data/chem/qm9/mingap.pkl \
            --n_retrain_epochs="$n_retrain_epochs" \
            --n_init_retrain_epochs="$n_init_retrain_epochs" \
            --n_best_points=2000 --n_rand_points=8000 \
            --n_inducing_points=5 \
            --invalid_score=-10000.0 \
            --weight_type="$weight_type" --rank_weight_k="$k"

    done
done