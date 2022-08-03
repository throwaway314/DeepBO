import numpy as np
import os
import pickle
import pandas

# qm9_smiles_path = os.path.expanduser("~/.spektral/datasets/qm9/qm9_smiles.npz")
# data = np.load(qm9_smiles_path, allow_pickle=True)
#
# smiles = data['smiles']
# y = data['y']

# print(smiles[22])
# print(y[1])

# with open('qm9_x.txt', 'w') as f:
#     for smile in smiles:
#         f.write("%s\n" % smile)

# with open('alphagap.txt', 'w') as f:
#     for yi in y[2:]:
#         f.write("%f\n" % (yi[5]+yi[8]))

# data = {}
# for (xi, yi) in zip(smiles, y):
#     # data[xi] = yi[5]
#     data[xi] = (yi[5]-6) / 191 - (yi[8] - 0.02) / 0.6
#     print(yi[5])
#     if yi[5] < small:
#         small = yi[5]


from rdkit import Chem
from rdkit.Chem.PandasTools import LoadSDF
# sdf_filename = os.path.expanduser("~/.spektral/datasets/qm9/qm9.sdf")
sdf_filename = "data/gdb9.sdf"
df = LoadSDF(sdf_filename, smilesName='SMILES')
# print(df.loc[:, 'SMILES'])
# # print(df.loc[:, 'SMILES'].tolist())
# for col in df.columns:
#     print(col)
print(df.shape)     # 133247, 3
smiles = df.loc[:, 'SMILES'].tolist()

df2 = pandas.read_csv(os.path.expanduser("~/.spektral/datasets/qm9/qm9.sdf.csv"))
# for col in df2.columns:
#     print(col)

# j = 0
# for i in range(133885):
# # for i in range (5):
#     suppl = Chem.SDMolSupplier(os.path.expanduser(f"~/.spektral/datasets/qm9/separate/{i}.sdf"))
#     for mol in suppl:
#         if mol is None:
#             j += 1
# # print((i, smile))
# print(j)

# i = 0
# j = 0
# for smile in smiles:
#     i += 1
#     m = Chem.MolFromSmiles(smile)
#     if m is None:
#         j += 1
        # print((i, smile))

data_alpha = {}
data_gap = {}
for index, row in df2.iterrows():
    data_alpha[row['mol_id']] = row['alpha']
    data_gap[row['mol_id']] = row['gap']

data2_alpha = {}
data2_mingapalpha = {}
data2_gap = {}
for index, row in df.iterrows():
    smile = row['SMILES']
    value_alpha = data_alpha[row['ID']]
    value_gap = data_gap[row['ID']]
    data2_alpha[smile] = value_alpha
    data2_mingapalpha[smile] = (value_alpha-6) / 191 - (value_gap - 0.02) / 0.6
    data2_gap[smile] = -value_gap
print(len(data2_alpha))

with open('data/chem/qm9/qm9_smiles.txt', 'w') as f:
    for smile in data2_alpha.keys():
        f.write("%s\n" % smile)
with open('data/chem/qm9/alpha.pkl', 'wb') as f:
    pickle.dump(data2_alpha, f)
with open('data/chem/qm9/mingapalpha.pkl', 'wb') as f:
    pickle.dump(data2_mingapalpha, f)
with open('data/chem/qm9/mingap.pkl', 'wb') as f:
    pickle.dump(data2_gap, f)

