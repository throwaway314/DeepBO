
# Copy stored data to correct folder
chem_dir="data/chem/qm9"
mkdir -p "$chem_dir"

# Store directories to the correct scripts
preprocess_script="weighted_retraining/chem/preprocess_data.py"


# Normally you might make the vocab, but the vocab is already made
# (it was copied from the original repo,
# so may not be exactly reproducible with the code in this repo)
# To make vocab for another model/dataset, run a command like the following:
# python scripts/data/chem/create_vocab.py \
#   --input_file=data/chem/zinc/orig_model/train.txt \
#   --output_file=data/chem/zinc/orig_model/vocab-CHECK.txt


# Next, we preprocess all the data (takes a VERY long time sadly...)

# Training set
out_dir="$chem_dir"/tensors_train
mkdir "$out_dir"
python "$preprocess_script" \
    -t "$chem_dir"/qm9_smiles.txt \
    -d "$out_dir"
