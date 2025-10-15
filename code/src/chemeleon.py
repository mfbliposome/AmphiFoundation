from datetime import datetime
import os
from matplotlib import pyplot as plt
import numpy as np
# from sklearn.base import r2_score
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from rdkit.Chem import MolFromSmiles
from chemprop import featurizers, nn
from chemprop.data import BatchMolGraph
from chemprop.models import MPNN
from chemprop.nn import RegressionFFN

@torch.no_grad()
def chemeleon_latent_space(smiles: list[str], model_path: str = "chemeleon_mp.pt") -> torch.Tensor:
    # Load model checkpoint
    checkpoint = torch.load(model_path, weights_only=True)

    # Create components
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    agg = nn.MeanAggregation()
    mp = nn.BondMessagePassing(**checkpoint['hyper_parameters'])
    mp.load_state_dict(checkpoint['state_dict'])

    # Initialize model
    model = MPNN(
        message_passing=mp,
        agg=agg,
        predictor=RegressionFFN() 
    )
    model.eval()

    # Convert SMILES to batch graph
    mol_graphs = [featurizer(MolFromSmiles(s)) for s in smiles]
    bmg = BatchMolGraph(mol_graphs)

    # Move to correct device
    bmg.to(device=model.device)

    # Return fingerprint as numpy array
    return model.fingerprint(bmg).numpy(force=True)

# Usage example
# latent_space = chemeleon_latent_space(["C", "CC"])

def train_regressor(df_total):
    time_str = datetime.now().strftime('%Y%m%d_%H')
    save_dir = f'../results/{property}_{time_str}'
    os.makedirs(save_dir, exist_ok=True)
    
    property_='pCMC'
    train_df, test_df = train_test_split(
            df_total,
            test_size=0.2, 
            stratify=df_total['Surfactant_Type'],
            random_state=42  # for reproducibility
        )

    train_x = train_df.iloc[:,0:300]
    train_y = train_df[property_]

    test_x = test_df.iloc[:,0:300]
    test_y = test_df[property_]

    # regressor = SVR(kernel="rbf", degree=3, C=5, gamma="scale", epsilon=0.01)
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    model = TransformedTargetRegressor(regressor=regressor,
                                    transformer=MinMaxScaler(feature_range=(-1, 1))
                                    ).fit(train_x, train_y)

    pred_y = model.predict(test_x)
    RMSE_score = np.sqrt(mean_squared_error(test_y, pred_y))
    r2 = r2_score(test_y, pred_y)

    print(f"RMSE score : {RMSE_score}" )
    print(f"R2 score : {r2}" )

    # Scatter plot: True vs Predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(test_y, pred_y, color='blue', alpha=0.6, edgecolor='k')
    plt.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'r--', lw=2)  # y = x line
    plt.xlabel(f'True {property}', fontsize=14)
    plt.ylabel(f'Predicted {property}', fontsize=14)
    # plt.title('SVR Prediction vs True Values', fontsize=16)
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(save_dir, f'prediction_{property}_{time_str}_chemeleon.png')
    plt.savefig(plot_path, dpi=300)
    # fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def build_mixture_latent_features(df: pd.DataFrame, smi_cols: list[str], conc_cols: list[str], target_col: str, latent_fn) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a mixture dataframe with SMILES and concentrations, compute mixture latent representations.
    
    Parameters:
    - df: Input dataframe.
    - smi_cols: List of column names for component SMILES.
    - conc_cols: List of column names for corresponding concentrations.
    - target_col: Column name for target values (e.g., miscibility).
    - latent_fn: Function that takes a list of SMILES and returns their latent vectors.
    
    Returns:
    - x_latent: (n_samples, latent_dim) numpy array of mixture features.
    - y: (n_samples,) target values.
    """
    assert len(smi_cols) == len(conc_cols), "Mismatch between SMILES and concentration columns"
    
    # Get all unique SMILES from all smi columns
    all_smis = pd.unique(df[smi_cols].values.ravel()).tolist()
    
    # Get latent space lookup dictionary
    latent_lookup = dict(zip(all_smis, latent_fn(all_smis)))
    
    # Replace SMILES with latent vectors
    df_embed = df.copy()
    for smi_col in smi_cols:
        df_embed[f'{smi_col}_vec'] = df_embed[smi_col].map(lambda x: latent_lookup.get(x))

    # Drop rows with any NaN embeddings
    vec_cols = [f'{s}_vec' for s in smi_cols]
    df_embed = df_embed.dropna(subset=vec_cols).reset_index(drop=True)

    # Compute mixture feature vector
    def mix_row(row):
        vectors = [np.array(row[f'{s}_vec']) * row[c] for s, c in zip(smi_cols, conc_cols)]
        return sum(vectors)

    df_embed['mixture_vec'] = df_embed.apply(mix_row, axis=1)

    x_latent = np.stack(df_embed['mixture_vec'].values)
    y = df_embed[target_col] # series object
    
    return x_latent, y

# Usage:
# x_latent, y = build_mixture_latent_features(
#     df=df_mixture_norm,
#     smi_cols=['smi1', 'smi2'],
#     conc_cols=['conc1', 'conc2'],
#     target_col='miscibility',
#     latent_fn=chemeleon_latent_space
# )