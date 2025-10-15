from typing import List
from typing import Tuple
import numpy as np
import torch
import pandas as pd

# def latent_fn_VICGAE(smis: List[str], model) -> np.ndarray:
#     """
#     Given a list of SMILES strings, return their latent embeddings using a model.

#     Parameters:
#     - smis: List of SMILES strings.
#     - model: A model with .embed_smiles(smiles: str) -> torch.Tensor

#     Returns:
#     - embeddings: np.ndarray of shape (n_samples, latent_dim)
#     """
#     # filter disconnected mols
#     def keep_largest_fragment(smi):
#         return max(smi.split('.'), key=len)

#     # Clean SMILES: strip whitespace, remove empty/null
#     smis_clean = [
#         keep_largest_fragment(s.strip())
#         for s in smis
#         if isinstance(s, str) and s.strip()
#     ]
#     # smis_clean = [keep_largest_fragment(s) for s in smis]  
#     embeddings = torch.stack([model.embed_smiles(s) for s in smis_clean])
#     return embeddings.numpy().squeeze(1)  # shape: (n, latent_dim)

# def latent_fn_VICGAE(smis: List[str], model) -> np.ndarray:
#     def keep_largest_fragment(smi):
#         return max(smi.split('.'), key=len)

#     smis_clean = [
#         keep_largest_fragment(s.strip())
#         for s in smis
#         if isinstance(s, str) and s.strip()
#     ]

#     valid_embeddings = []
#     for s in smis_clean:
#         try:
#             emb = model.embed_smiles(s)
#             valid_embeddings.append(emb)
#         except KeyError as e:
#             print(f"⚠️ Skipping SMILES with unsupported token: {s} ({e})")

#     if not valid_embeddings:
#         raise ValueError("No valid SMILES found after filtering.")

#     return torch.stack(valid_embeddings).numpy().squeeze(1)

def latent_fn_VICGAE(smis: List[str], model) -> np.ndarray:
    def keep_largest_fragment(smi):
        return max(smi.split('.'), key=len)

    smis_clean = [
        keep_largest_fragment(s.strip())
        for s in smis
        if isinstance(s, str) and s.strip()
    ]

    valid_embeddings = []
    skipped = []

    for s in smis_clean:
        try:
            emb = model.embed_smiles(s)
            valid_embeddings.append(emb)
        except Exception as e:
            skipped.append((s, str(e)))

    # Print all skipped SMILES at once
    if skipped:
        print(f"\nSkipped {len(skipped)} SMILES due to errors:")
        for smi, err in skipped:
            print(f"  - {smi}: {err}")

    if not valid_embeddings:
        raise ValueError("No valid SMILES found after filtering.")

    return torch.stack(valid_embeddings).numpy().squeeze(1)




def build_mixture_latent_features_VICGAE(
    df: pd.DataFrame,
    smi_cols: List[str],
    conc_cols: List[str],
    target_col: str,
    latent_fn,  # should be a callable like latent_fn_from_model
    latent_fn_args: dict = None  # any args to pass to latent_fn (e.g., {"model": my_model})
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a mixture dataframe with SMILES and concentrations, compute mixture latent representations.
    
    Parameters:
    - df: Input dataframe.
    - smi_cols: List of column names for component SMILES.
    - conc_cols: List of column names for corresponding concentrations.
    - target_col: Column name for target values (e.g., miscibility).
    - latent_fn: Function that takes a list of SMILES and returns latent vectors.
    - latent_fn_args: Dictionary of additional keyword arguments for latent_fn
    
    Returns:
    - x_latent: (n_samples, latent_dim) numpy array of mixture features.
    - y: (n_samples,) target values.
    """
    assert len(smi_cols) == len(conc_cols), "Mismatch between SMILES and concentration columns"

    # Get all unique SMILES from the mixture
    all_smis = pd.unique(df[smi_cols].values.ravel()).tolist()

    # Compute latent vectors using provided function
    latent_vectors = latent_fn(all_smis, **(latent_fn_args or {}))
    latent_lookup = dict(zip(all_smis, latent_vectors))

    # Replace SMILES with latent vectors
    df_embed = df.copy()
    for smi_col in smi_cols:
        df_embed[f'{smi_col}_vec'] = df_embed[smi_col].map(lambda x: latent_lookup.get(x))

    # Drop rows with any missing vectors
    vec_cols = [f'{s}_vec' for s in smi_cols]
    df_embed = df_embed.dropna(subset=vec_cols).reset_index(drop=True)

    # Combine latent vectors weighted by concentration
    def mix_row(row):
        return sum(np.array(row[f'{s}_vec']) * row[c] for s, c in zip(smi_cols, conc_cols))

    df_embed['mixture_vec'] = df_embed.apply(mix_row, axis=1)
    
    x_latent = np.stack(df_embed['mixture_vec'].values)
    y = df_embed[target_col].values  # as np.ndarray
    
    return x_latent, y
