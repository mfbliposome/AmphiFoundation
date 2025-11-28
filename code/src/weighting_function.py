import numpy as np
import pandas as pd

from VICGAE_latent import build_mixture_latent_features_VICGAE, latent_fn_VICGAE
from chemprop_latent import build_mixture_latent_features_chemprop, chemprop_latent_space
from chemeleon import build_mixture_latent_features, chemeleon_latent_space
from utils import binarize_last_column
from baseline import smiles_to_fp

def prepare_structured_df(
        df_input_update_ori,
        df_info,
        transform="log1p"
    ):
    """
    transform options:
        - "original"      : use raw concentrations
        - "log1p"         : apply log1p to mM concentration
        - "mole_fraction" : convert concentration to mole fraction
    """
    
    # select concentration columns
    df_input = df_input_update_ori.copy()
    conc = df_input.iloc[:, :7]
    
    # apply transformation
    if transform == "original":
        df_trans = conc.copy()

    elif transform == "log1p":
        df_trans = conc.applymap(lambda x: np.log1p(x))

    elif transform == "mole_fraction":
        eps = 1e-12
        total = conc.sum(axis=1) + eps
        df_trans = conc.divide(total, axis=0)

    else:
        raise ValueError("transform must be one of: 'original', 'log1p', 'mole_fraction'")
    
    # Replace concentration columns
    df_input.iloc[:, :7] = df_trans
    df_input = df_input.iloc[:, 0:8]  # keep 7 features + vesicle result
    
    # rename columns (remove '_Concentration (mM)')
    df = df_input.copy()
    df.columns = [col.replace('_Concentration (mM)', '') 
                  for col in df.columns[:-1]] + ['num_vesicles']
    
    #  mapping from column name → amphiphile long name
    column_to_name = {
        'decanoic acid': 'Decanoic acid',
        'decanoate': 'Decanoate',
        'decylamine': 'Decylamine',
        'decyl trimethylamine': 'Decyltrimethyl ammonium bromid',
        'decylsulfate': 'Decyl sodium sulfate',
        'decanol': 'Decanol',
        'monocaprin': 'Glycerol monodecanoate'
    }

    # build mapping to SMILES from df_info
    name_to_smiles = dict(zip(df_info['Name'], df_info['SMILES']))
    
    # construct structured dataframe
    new_data = {}

    for col in df.columns[:-1]:  # skip num_vesicles
        chem_name = column_to_name[col]
        smiles = name_to_smiles.get(chem_name, '')
        new_data[f"{col}_SMILES"] = [smiles] * len(df)
        new_data[f"{col}_Concentration"] = df[col]

    new_data["num_vesicles"] = df["num_vesicles"]
    
    df_structured = pd.DataFrame(new_data)
    
    # rename 
    final_names = [
        'smi1', 'conc1',
        'smi2', 'conc2',
        'smi3', 'conc3',
        'smi4', 'conc4',
        'smi5', 'conc5',
        'smi6', 'conc6',
        'smi7', 'conc7',
        'vesicles_formation'
    ]
    
    df_structured.columns = final_names
    
    return df_structured


def prepare_latent_dataset_am(df_structured, latent_fn):
    """
    df_structured: dataframe (with smi1…smi7 and conc1…conc7 + vesicles_formation)
    and returns a latent-space dataframe with the binarized vesicle label.

    latent_fn: a function that computes latent space given.

    """
    
    # convert SMILES to latent vectors
    X_smi = latent_fn(df_structured)
    
    # extract vesicle labels
    y_smi = df_structured['vesicles_formation']
    
    # combine latent vectors with labels
    df_total = pd.concat([X_smi, y_smi], axis=1)
    
    # binarize the last column
    df_total_cls = binarize_last_column(df_total)
    
    return df_total_cls

def prepare_latent_dataset_baseline(df_structured):
    """
    Prepare latent dataset with fingerprint baseline model

    """
    for i in range(1, 8):
        col_name = f'smi{i}'
        df_structured[col_name] = df_structured[col_name].apply(smiles_to_fp)

    def build_feature_vector(df, smi_cols, conc_cols):
        components = [df[smi].apply(pd.Series).mul(df[conc], axis=0) for smi, conc in zip(smi_cols, conc_cols)]
        return sum(components)
    
    df_structured_cls = binarize_last_column(df_structured)

    smi_cols = [f'smi{i}' for i in range(1, 8)]
    conc_cols = [f'conc{i}' for i in range(1, 8)]

    X = build_feature_vector(df_structured, smi_cols, conc_cols)
    y = df_structured_cls['vesicles_formation'].values
    
    df_stable=pd.concat([X, df_structured_cls['vesicles_formation']], axis=1)

    
    return df_stable


def prepare_latent_dataset_chemprop(df_structured):
    """
    Prepare latent dataset with chemprop model

    """
    df_structured_classify = binarize_last_column(df_structured)

    checkpoint_path='../../src/models/chemprop/example_model_v2_regression_mol.ckpt'

    x_latent_am, y_am = build_mixture_latent_features_chemprop(
        df=df_structured_classify,
        smi_cols=['smi1', 'smi2', 'smi3', 'smi4', 'smi5', 'smi6', 'smi7'],
        conc_cols=['conc1', 'conc2', 'conc3', 'conc4', 'conc5', 'conc6', 'conc7'],
        target_col='vesicles_formation',
        latent_fn=chemprop_latent_space,
        latent_fn_args={"checkpoint_path": checkpoint_path}
    )

    df_total = pd.concat([pd.DataFrame(x_latent_am), pd.DataFrame(y_am, columns=['vesicles_formation'])],axis=1)

    
    return df_total

def prepare_latent_dataset_chemeleon(df_structured):
    """
    Prepare latent dataset with chemeleon model
    """
    df_structured_classify = binarize_last_column(df_structured)

    x_latent, y = build_mixture_latent_features(
    df=df_structured_classify,
    smi_cols=['smi1', 'smi2', 'smi3', 'smi4', 'smi5', 'smi6', 'smi7'],
    conc_cols=['conc1', 'conc2', 'conc3', 'conc4', 'conc5', 'conc6', 'conc7'],
    target_col='vesicles_formation',
    latent_fn=chemeleon_latent_space)

    df_total = pd.concat([pd.DataFrame(x_latent), y],axis=1)
    
    return df_total