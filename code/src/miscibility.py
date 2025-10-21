import pandas as pd
import numpy as np
import torch
from models.smi_ted.smi_ted_light.load import load_smi_ted


def calculate_mM(concentration_percent, monomer_mw):
    """
    concentration_percent: % w/w (grams solute per 100 g solution)
    monomer_mw: g/mol (monomer molecular weight)

    returns: concentration in mM in final 20 µL solution (after adding 10 µL molecule solution to 10 µL other solution)
    """
    
    if pd.isna(concentration_percent) or pd.isna(monomer_mw) or monomer_mw == 0:
        return np.nan

    # grams of solute in 10 µL
    grams_per_100ml = concentration_percent / 100  # g/mL
    volume_added_ml = 10 / 1000  # 10 µL → mL
    grams_solute = grams_per_100ml * volume_added_ml

    # moles of solute
    moles_solute = grams_solute / monomer_mw

    # calculate molarity in final 20 µL (0.00002 L)
    total_volume_L = 20 / 1_000_000  # 20 µL → L
    molarity_M = moles_solute / total_volume_L

    # convert to mM
    molarity_mM = molarity_M * 1000

    return molarity_mM

def is_symmetric(matrix):
    """
    Checks if a square matrix is symmetric.

    Args:
        matrix (list of lists or numpy.ndarray): The matrix to check.

    Returns:
        bool: True if the matrix is symmetric, False otherwise.
    """
    # Ensure it's a NumPy array for easier manipulation
    matrix = np.array(matrix)

    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False  # A non-square matrix cannot be symmetric

    # Check for symmetry: compare the matrix to its transpose
    return np.allclose(matrix, matrix.T) 


def get_latent_space_m(df):

    train_df = df.copy()

    train_smiles_list = pd.concat([train_df[f'smi{i}'] for i in range(1, 3)]).unique().tolist()
    
    model_SMI = load_smi_ted(folder='../src/models/smi_ted/smi_ted_light', ckpt_filename='smi-ted-Light_40.pt')
    
    return_tensor=True
    with torch.no_grad():
        x_emb = model_SMI.encode(train_smiles_list, return_torch=return_tensor)
    
    x_emb_array  = x_emb.numpy()
    x_emb_frame = pd.DataFrame(x_emb_array)

    train_emb = [np.nan if row.isna().all() else row.dropna().tolist() for _, row in x_emb_frame.iterrows()]
    
    train_dict = dict(zip(train_smiles_list, train_emb))
    
    def replace_with_list(value, my_dict):
        return my_dict.get(value, value)
    
    # Replace the smiles string with its embeddings
    df_train_emb = train_df.applymap(lambda x: replace_with_list(x, train_dict))
    
    # Drop rows with NaN and reset index
    df_train_emb = df_train_emb.dropna().reset_index(drop=True)
    
    # Normalize concentrations
    conc_cols = [f'conc{i}' for i in range(1, 3)]
    df_train_emb = normalize_concentrations(df_train_emb, conc_cols)
    
    # Construct latent space for mixture
    def build_feature_vector(df, smi_cols, conc_cols):
        components = [df[smi].apply(pd.Series).mul(df[conc], axis=0) for smi, conc in zip(smi_cols, conc_cols)]
        return sum(components)
    
    smi_cols = [f'smi{i}' for i in range(1, 3)]
    conc_cols = [f'conc{i}' for i in range(1, 3)]
    
    x_smi = build_feature_vector(df_train_emb, smi_cols, conc_cols)

    return df_train_emb, x_smi

def normalize_concentrations(df, conc_cols):
    conc_sum = df[conc_cols].sum(axis=1)
    for conc in conc_cols:
        df[conc] = df[conc] / conc_sum
    return df

def normalize_concentrations_log(df, conc_cols):
    for conc in conc_cols:
        df[conc] = np.log1p(df[conc])  
    return df

def get_latent_space_m_log(df):
    train_df = df.copy()

    train_smiles_list = pd.concat([train_df[f'smi{i}'] for i in range(1, 3)]).unique().tolist()
    
    model_SMI = load_smi_ted(folder='../src/models/smi_ted/smi_ted_light', ckpt_filename='smi-ted-Light_40.pt')
    
    return_tensor=True
    with torch.no_grad():
        x_emb = model_SMI.encode(train_smiles_list, return_torch=return_tensor)
    
    x_emb_array  = x_emb.numpy()
    x_emb_frame = pd.DataFrame(x_emb_array)

    train_emb = [np.nan if row.isna().all() else row.dropna().tolist() for _, row in x_emb_frame.iterrows()]
    
    train_dict = dict(zip(train_smiles_list, train_emb))
    
    def replace_with_list(value, my_dict):
        return my_dict.get(value, value)
    
    # Replace the smiles string with its embeddings
    df_train_emb = train_df.applymap(lambda x: replace_with_list(x, train_dict))
    
    # Drop rows with NaN and reset index
    df_train_emb = df_train_emb.dropna().reset_index(drop=True)
    
    # Normalize concentrations
    conc_cols = [f'conc{i}' for i in range(1, 3)]
    df_train_emb = normalize_concentrations_log(df_train_emb, conc_cols)
    
    # Construct latent space for mixture
    def build_feature_vector(df, smi_cols, conc_cols):
        components = [df[smi].apply(pd.Series).mul(df[conc], axis=0) for smi, conc in zip(smi_cols, conc_cols)]
        return sum(components)
    
    smi_cols = [f'smi{i}' for i in range(1, 3)]
    conc_cols = [f'conc{i}' for i in range(1, 3)]
    
    x_smi = build_feature_vector(df_train_emb, smi_cols, conc_cols)

    return df_train_emb, x_smi
