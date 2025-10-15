
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import RDKFingerprint
from rdkit.DataStructs import ConvertToNumpyArray

def get_rdkit_descriptors_v1(df_data ):

    # Convert SMILES to RDKit Mol objects
    df_data['mol'] = df_data['SMILES'].apply(Chem.MolFromSmiles)
    # Drop rows where mol conversion failed
    df_data = df_data[df_data['mol'].notnull()].reset_index(drop=True)
    fp_array = np.array([mol_to_rdkfp(mol) for mol in df_data['mol']])
    # Create DataFrame of fingerprint features
    df_fp = pd.DataFrame(fp_array)
    df_fp.columns = [f'RDKFP_{i}' for i in range(df_fp.shape[1])]
    df_fp['pCMC'] = df_data['pCMC'].values   

    return df_fp

def smiles_to_fp(smiles, fp_size=2048):
    """Convert SMILES to RDKFingerprint array"""
    mol = Chem.MolFromSmiles(smiles)
    arr = np.zeros((fp_size,), dtype=int)
    if mol is not None:
        fp = RDKFingerprint(mol, fpSize=fp_size)
        ConvertToNumpyArray(fp, arr)
    return arr

# def get_rdkit_descriptors_v2(df_data ):
#     smiles = df_data['SMILES']

#     # Generate fingerprints
#     fps = [RDKFingerprint(Chem.MolFromSmiles(s)) for s in smiles]

#     # Convert to numpy arrays
#     fp_array = []
#     for fp in fps:
#         arr = np.zeros((1,))
#         ConvertToNumpyArray(fp, arr)
#         fp_array.append(arr)

#     # Create DataFrame
#     df_fp = pd.DataFrame(fp_array)
#     df_fp.columns = [f'RDKFP_{i}' for i in range(df_fp.shape[1])]

#     # Add target
#     df_fp['pCMC'] = df_data['pCMC'].values  

#     return df_fp      

# Generate RDK Fingerprints
def mol_to_rdkfp(mol, fp_size=2048):
    fp = RDKFingerprint(mol, fpSize=fp_size)
    arr = np.zeros((1,))
    ConvertToNumpyArray(fp, arr)
    return arr

def build_feature_vector_bm(df, smi_cols, conc_cols):
    features = []
    for idx, row in df.iterrows():
        # Ensure weighted_fp is float type to allow float additions
        weighted_fp = np.zeros_like(row[smi_cols[0]], dtype=np.float64)
        for smi_col, conc_col in zip(smi_cols, conc_cols):
            weighted_fp += row[smi_col] * row[conc_col]
        features.append(weighted_fp)
    return np.array(features)



# def build_feature_vector_bm(df, smi_cols, conc_cols):
#     features = []
#     for idx, row in df.iterrows():
#         weighted_fp = None
#         for smi_col, conc_col in zip(smi_cols, conc_cols):
#             smi_array = np.array(row[smi_col])  # Convert to NumPy array
#             weighted = smi_array * row[conc_col]  # Element-wise multiplication
#             if weighted_fp is None:
#                 weighted_fp = weighted
#             else:
#                 weighted_fp += weighted
#         features.append(weighted_fp)
#     return np.array(features)
