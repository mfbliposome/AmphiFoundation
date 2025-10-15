import pandas as pd
from chemprop import data, models, featurizers
import torch
import numpy as np
from rdkit import Chem

@torch.no_grad()
def chemprop_latent_space(smiles: list[str], checkpoint_path: str) -> np.ndarray:
    """
    Given a list of SMILES strings, return their latent embeddings using a Chemprop model.

    Parameters:
    - smiles: List of SMILES strings.
    - checkpoint_path: Path to the Chemprop checkpoint file.

    Returns:
    - embeddings: np.ndarray of shape (n_samples, latent_dim)
    """
    # Load model
    model = models.MPNN.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Featurizer
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    # Filter and featurize
    clean_smis = [s for s in smiles if isinstance(s, str) and s.strip()]
    datapoints = [data.MoleculeDatapoint.from_smi(s) for s in clean_smis]
    dataset = data.MoleculeDataset(datapoints, featurizer=featurizer)
    loader = data.build_dataloader(dataset, shuffle=False)

    # Collect encodings
    encodings = [
        model.encoding(batch.bmg, batch.V_d, batch.X_d, i=-1)
        for batch in loader
    ]
    encodings = torch.cat(encodings, dim=0)

    return encodings.numpy()


def build_mixture_latent_features_chemprop(
    df: pd.DataFrame,
    smi_cols: list[str],
    conc_cols: list[str],
    target_col: str,
    latent_fn,  # like chemprop_latent_space
    latent_fn_args: dict = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a mixture dataframe with SMILES and concentrations, compute mixture latent representations.

    Returns:
    - x_latent: (n_samples, latent_dim) numpy array of mixture features.
    - y: (n_samples,) target values.
    """
    assert len(smi_cols) == len(conc_cols), "Mismatch between SMILES and concentration columns"

    all_smis = pd.unique(df[smi_cols].values.ravel()).tolist()
    latent_vectors = latent_fn(all_smis, **(latent_fn_args or {}))
    latent_lookup = dict(zip(all_smis, latent_vectors))

    df_embed = df.copy()
    for smi_col in smi_cols:
        df_embed[f'{smi_col}_vec'] = df_embed[smi_col].map(lambda x: latent_lookup.get(x))

    vec_cols = [f'{s}_vec' for s in smi_cols]
    df_embed = df_embed.dropna(subset=vec_cols).reset_index(drop=True)

    def mix_row(row):
        vectors = [np.array(row[f'{s}_vec']) * row[c] for s, c in zip(smi_cols, conc_cols)]
        return sum(vectors)

    df_embed['mixture_vec'] = df_embed.apply(mix_row, axis=1)
    x_latent = np.stack(df_embed['mixture_vec'].values)
    y = df_embed[target_col].values

    return x_latent, y
