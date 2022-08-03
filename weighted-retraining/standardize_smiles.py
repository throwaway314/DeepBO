from weighted_retraining.chem.chem_utils import standardize_smiles

with open('qm9_smiles.txt') as f:
    lines = f.readlines()

smiles = lines
canonic_smiles = list(map(standardize_smiles, smiles))

with open('qm9_smiles_standardized.txt', 'w') as f:
    for item in canonic_smiles:
        f.write(f'{item}\n')

