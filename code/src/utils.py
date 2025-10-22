import joblib
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
from xgboost import XGBClassifier
from models.smi_ted.smi_ted_light.load import load_smi_ted
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px


from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split

def get_molecule_representation(df_structured=None, train_smiles_list=None):
    """
    Generates molecule representations using a SMILES embedding model.

    Args:
        df_structured (pd.DataFrame, optional): DataFrame containing SMILES strings
                                                 in columns 'smi1' to 'smi7'.
                                                 Required if train_smiles_list is None.
        train_smiles_list (list, optional): A list of SMILES strings.
                                             If provided, df_structured is ignored.

    Returns:
        pd.DataFrame: DataFrame containing the molecule embeddings.
    """
    model_SMI = load_smi_ted(folder='../src/models/smi_ted/smi_ted_light', ckpt_filename='smi-ted-Light_40.pt')

    if train_smiles_list is not None:
        return_tensor=True
        with torch.no_grad():
            x_emb = model_SMI.encode(train_smiles_list, return_torch=return_tensor)

        x_emb_array  = x_emb.numpy()
        x_emb_frame = pd.DataFrame(x_emb_array)
    elif df_structured is not None:
        train_smiles_list = pd.concat([df_structured[f'smi{i}'] for i in range(1, 8)]).unique().tolist()

        return_tensor=True
        with torch.no_grad():
            x_emb = model_SMI.encode(train_smiles_list, return_torch=return_tensor)

        x_emb_array  = x_emb.numpy()
        x_emb_frame = pd.DataFrame(x_emb_array)
    else:
            raise ValueError("Either df_structured or train_smiles_list must be provided.")

    return x_emb_frame


def get_latent_space(df):
    train_df = df.copy()

    train_smiles_list = pd.concat([train_df[f'smi{i}'] for i in range(1, 8)]).unique().tolist()
    
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
    
    # Construct latent space for mixture
    def build_feature_vector(df, smi_cols, conc_cols):
        components = [df[smi].apply(pd.Series).mul(df[conc], axis=0) for smi, conc in zip(smi_cols, conc_cols)]
        return sum(components)
    
    smi_cols = [f'smi{i}' for i in range(1, 8)]
    conc_cols = [f'conc{i}' for i in range(1, 8)]
    
    x_smi = build_feature_vector(df_train_emb, smi_cols, conc_cols)

    return x_smi

def run_classifier(xtrain, ytrain, xtest):

    # Train XGBoost classifier
    xgb_clf = XGBClassifier(n_estimators=5000, learning_rate=0.01, max_depth=10, use_label_encoder=False, eval_metric='logloss')
    xgb_clf.fit(xtrain, ytrain)

    # Predict probabilities for ROC-AUC
    # y_pred = xgb_clf.predict(xtest)
    y_pred = xgb_clf.predict_proba(xtest)[:,1]

    # ROC-AUC score
    roc_auc = roc_auc_score(ytrain, y_pred)
    fpr, tpr, threshold = roc_curve(ytrain, y_pred)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    return xgb_clf, y_pred, roc_auc, fpr, tpr, threshold


def run_classifier_update(xtrain, ytrain, xtest, ytest, classifier_alter=False):
    # Check if ytrain has both classes
    unique_classes = np.unique(ytrain)

    if len(unique_classes) < 2:
        only_class = unique_classes[0]
        print(f"[Info] Only one class ({only_class}) in ytrain. Skipping training, returning dummy predictions.")
        
        # All predicted probabilities are 0.0 or 1.0 based on the only class
        pred_prob = np.full(len(xtest), float(only_class))
        
        # Return dummy values for everything else
        return (
            None,                     # model
            pred_prob,                # y_pred
            np.nan,                   # roc_auc
            np.array([]),             # fpr
            np.array([]),             # tpr
            np.array([]),             # threshold
        )
    
    # Choose classifier
    if not classifier_alter:
        clf = XGBClassifier(
            n_estimators=5000,
            learning_rate=0.01,
            max_depth=10,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            random_state=42,
        )
    
    # Fit the chosen classifier
    clf.fit(xtrain, ytrain)

    # Predict probabilities for ROC-AUC
    y_pred = clf.predict_proba(xtest)[:, 1]

    # Compute ROC-AUC metrics
    roc_auc = roc_auc_score(ytest, y_pred)
    fpr, tpr, threshold = roc_curve(ytest, y_pred)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    return clf, y_pred, roc_auc, fpr, tpr, threshold

def plot_PCA(df_total):
    X = df_total.iloc[:, :768].values
    y = df_total['miscibility'].values

    pca = PCA(n_components=10)  
    X_pca = pca.fit_transform(X)

    explained_variance = pca.explained_variance_ratio_

    miscibility_types = df_total['miscibility'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(miscibility_types)))
    color_dict = dict(zip(miscibility_types, colors))

    plt.figure(figsize=(8,6))
    for surfactant in miscibility_types:
        idx = (y == surfactant)
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], 
                    color=color_dict[surfactant], 
                    label=surfactant, alpha=0.7, s=30)

    plt.xlabel(f"PC1 ({explained_variance[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({explained_variance[1]*100:.1f}%)")
    plt.title('PCA 2D Projection')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # 3D Interactive Scatter Plot with plotly
    traces = []
    for surfactant in miscibility_types:
        idx = (y == surfactant)
        trace = go.Scatter3d(
            x=X_pca[idx, 0],
            y=X_pca[idx, 1],
            z=X_pca[idx, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=f'rgb({color_dict[surfactant][0]*255},{color_dict[surfactant][1]*255},{color_dict[surfactant][2]*255})',
                line=dict(width=0.5, color='DarkSlateGrey')
            ),
            name=surfactant
        )
        traces.append(trace)

    layout = go.Layout(
        title='PCA 3D Projection',
        scene=dict(
            xaxis_title=f'PC1 ({explained_variance[0]*100:.1f}%)',
            yaxis_title=f'PC2 ({explained_variance[1]*100:.1f}%)',
            zaxis_title=f'PC3 ({explained_variance[2]*100:.1f}%)'
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(title='Surfactant Type')
    )

    fig = go.Figure(data=traces, layout=layout)
    pio.show(fig)

    # 6. Explained Variance Plot
    plt.figure(figsize=(6,4))
    sns.barplot(x=[f'PC{i+1}' for i in range(10)], y=explained_variance*100, color='skyblue')
    plt.ylabel('Explained Variance (%)')
    plt.title('PCA Explained Variance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return

def binarize_last_column(df):
    """
    Converts the last column of a DataFrame to binary values:
    1 if value > 0, else 0.
    Returns a modified copy of the DataFrame.
    """
    df_bin = df.copy()
    last_col = df_bin.columns[-1]
    df_bin[last_col] = (df_bin[last_col] > 0).astype(int)
    return df_bin

# def get_classifier(xtrain, ytrain):

#     # Train XGBoost classifier
#     xgb_clf = XGBClassifier(n_estimators=5000, learning_rate=0.01, max_depth=10, use_label_encoder=False, eval_metric='logloss')
#     xgb_clf.fit(xtrain, ytrain)
    
#     return xgb_clf

def reverse_log1p(transformed_value):
  
    """
    Reverses the log1p transformation to get the original value.

    Args:
        transformed_value: The value after the log1p transformation.

    Returns:
        The original value before the log1p transformation.
    """
    original_value = np.expm1(transformed_value)

    return original_value

def replace_entire_smi_column(df, smi_column, new_smi):
    """
    Replace the entire smi_column in df with a single new SMILES.

    Parameters:
    - df: pd.DataFrame — df_structured
    - smi_column: str — column to replace (e.g., "smi3")
    - new_smi: str — SMILES string to set for the whole column

    Returns:
    - df_new: pd.DataFrame — modified copy
    """
    df_new = df.copy()

    if smi_column not in df.columns:
        raise ValueError(f"{smi_column} not found in DataFrame columns.")
    
    df_new[smi_column] = new_smi

    return df_new

def repeated_roc_analysis(X, y, n_repeats=10, test_size=0.2, random_state=42):
    """
    Repeated runs, plots mean ROC curve with shaded variance.

    Parameters:
    - X, y: dataset
    - n_repeats: int — how many repeats (default=10)
    - test_size: float — test size fraction (default=0.2)
    - random_state: int — seed (for reproducibility)

    Returns:
    - None (plots directly)
    """

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    rng = np.random.default_rng(seed=random_state)

    for i in range(n_repeats):
        rs = rng.integers(0, 10000)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=rs, stratify=y)

        def get_classifier(X_train, y_train):
            rf_clf = RandomForestClassifier(random_state=42)
            rf_clf.fit(X_train, y_train)
            return rf_clf

        rf_clf = get_classifier(X_train, y_train)
        model = rf_clf

        y_prob_test = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob_test)
        auc_score = roc_auc_score(y_test, y_prob_test)

        # Interpolate tpr to common fpr
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0  # force 0 at start
        tprs.append(interp_tpr)
        aucs.append(auc_score)

    # Compute mean + std
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_tpr[-1] = 1.0  # force end at 1

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    # Plot
    plt.figure(figsize=(7,6))
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})',
             lw=2, alpha=1)

    plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                     color='blue', alpha=0.2, label='±1 std dev')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Repeated ROC ({n_repeats} repeats)')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def evaluate_model_stability(df, n_repeats=10, random_state=42, property='vesicles_formation'):
    """
    Repeated ROC analysis to assess overfitting and stability.
    Plots all ROC curves + mean + shaded region.
    
    Parameters:
    - df: pd.DataFrame — your input dataset (df1)
    - n_repeats: int — number of runs (default=10)
    - random_state: int — reproducibility
    - property: output type

    Returns:
    - None (plots directly)
    """
    X = get_latent_space(df)
    y = df[property]

    rng = np.random.default_rng(seed=random_state)

    # Store interpolated TPRs + AUCs
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    plt.figure(figsize=(8,6))

    for i in range(n_repeats):
        # Train/test split (different random_state each time)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                             random_state=rng.integers(0, 10000),
                                                             stratify=y)
        # Train classifier
        model_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        model_clf.fit(X_train, y_train)

        # Predict on test set
        y_pred_test = model_clf.predict_proba(X_test)[:, 1]

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_test)
        roc_auc = auc(fpr, tpr)

        # Interpolate TPR over mean_fpr grid
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0  # Ensure starts at 0
        tprs.append(tpr_interp)
        aucs.append(roc_auc)

        # Plot individual ROC curve
        plt.plot(fpr, tpr, lw=1, alpha=0.6, label=f'Run {i+1} (AUC={roc_auc:.2f})')

    # Compute mean + std of TPRs
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)

    # Plot mean ROC + shaded area
    plt.plot(mean_fpr, mean_tpr, color='black', lw=2, label=f'Mean ROC (AUC={mean_auc:.2f})')
    plt.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0), 
                     np.minimum(mean_tpr + std_tpr, 1), color='grey', alpha=0.2,
                     label='±1 std dev')

    # Random guess baseline
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Repeated ROC ({n_repeats} repeats)')
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f'Mean AUC over {n_repeats} runs: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}')

def evaluate_model_stability_baseline(df, n_repeats=10, random_state=42):
    """
    Repeated ROC analysis to assess overfitting and stability.
    Plots all ROC curves + mean + shaded region.
    
    Parameters:
    - df: pd.DataFrame — your input dataset (df1)
    - n_repeats: int — number of runs (default=10)
    - random_state: int — reproducibility

    Returns:
    - None (plots directly)
    """
    X = df.iloc[:, 0:-1]
    y = df.iloc[:, -1:]

    rng = np.random.default_rng(seed=random_state)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    plt.figure(figsize=(8,6))

    for i in range(n_repeats):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                             random_state=rng.integers(0, 10000),
                                                             stratify=y)
        # Train classifier
        model_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        model_clf.fit(X_train, y_train)

        # Predict on test set
        y_pred_test = model_clf.predict_proba(X_test)[:, 1]

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_test)
        roc_auc = auc(fpr, tpr)

        # Interpolate TPR over mean_fpr grid
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0  # Ensure starts at 0
        tprs.append(tpr_interp)
        aucs.append(roc_auc)

        # Plot individual ROC curve
        plt.plot(fpr, tpr, lw=1, alpha=0.6, label=f'Run {i+1} (AUC={roc_auc:.2f})')

    # Compute mean + std of TPRs
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)

    # Plot mean ROC + shaded area
    plt.plot(mean_fpr, mean_tpr, color='black', lw=2, label=f'Mean ROC (AUC={mean_auc:.2f})')
    plt.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0), 
                     np.minimum(mean_tpr + std_tpr, 1), color='grey', alpha=0.2,
                     label='±1 std dev')

    # Random guess baseline
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Repeated ROC ({n_repeats} repeats)')
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f'Mean AUC over {n_repeats} runs: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}')

def normalize_concentrations(df, conc_cols):
    '''
    Normalize concentration ratio
    '''
    conc_sum = df[conc_cols].sum(axis=1)
    for conc in conc_cols:
        df[conc] = df[conc] / conc_sum
    return df


# ===========================================
# This part is for generate experimental plan

def pca_model_contour_plot(X, y_pred, model, n_components=2, grid_step=0.1,
                            cmap='viridis', levels=20, figsize=(8,6), plot_title=True,
                            random_state=42):
    """
    Reduce X to 2D using PCA, predict model on grid, and plot contour + scatter.

    Parameters:
    - X: np.array or pd.DataFrame (n_samples, n_features)
    - y_pred: array-like (n_samples,) — predicted probability or label for scatter color
    - model: trained classifier with predict_proba()
    - n_components: int — PCA components (default=2)
    - grid_step: float — step size of grid (default=0.1)
    - cmap: str — colormap (default='viridis')
    - levels: int — contour levels (default=20)
    - figsize: tuple — figure size (default=(8,6))
    - plot_title: str — title of the plot
    - random_state: int - random seed

    Returns:
    - None (plots directly)
    """
    # PCA reduction
    pca = PCA(n_components=n_components, random_state=random_state)
    X_reduced = pca.fit_transform(X)
    Z_prob_ori = model.predict_proba(X)[:, 1]


    explained_var = pca.explained_variance_ratio_
    print(f"Explained variance ratio (first {n_components} components): {explained_var}")

    # Mesh grid in PCA space
    x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
    y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                         np.arange(y_min, y_max, grid_step))

    # Inverse transform grid to original space
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    X_grid_original = pca.inverse_transform(grid_points)

    # Predict probability on grid
    Z_prob = model.predict_proba(X_grid_original)[:, 1]
    Z_prob = Z_prob.reshape(xx.shape)

    # Plot contour + scatter
    plt.figure(figsize=figsize)
    contour = plt.contourf(xx, yy, Z_prob, levels=levels, cmap=cmap)
    plt.colorbar(contour, label='Predicted Probability')

    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_pred, edgecolor='k', cmap=cmap, s=20)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    if plot_title:
        plot_title_str='PCA + Contour plot (Predicted Probability)'
        plt.title(plot_title_str)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return pca, grid_points, Z_prob_ori, X_reduced

def select_diverse_grid_points(pca, grid_points, Z_prob, n_high=10, n_low=10, threshold_high=0.8, threshold_low=0.2):
    """
    Select diverse grid points from high and low probability regions.

    Parameters:
    - pca: fitted PCA object
    - grid_points: (N, 2) points in PCA space (meshgrid flattened)
    - Z_prob: (N,) predicted probabilities at grid points
    - n_high: number of diverse high-probability points to select
    - n_low: number of diverse low-probability points to select
    - threshold_high: minimum prob for high selection
    - threshold_low: maximum prob for low selection

    Returns:
    - high_PCA, low_PCA: selected points in PCA space
    - high_original, low_original: selected points in original feature space
    """
    # Identify high and low prob regions
    high_mask = Z_prob >= threshold_high
    low_mask = Z_prob <= threshold_low

    high_candidates = grid_points[high_mask]
    low_candidates = grid_points[low_mask]

    def farthest_point_sampling(X_subset, k):
        if len(X_subset) == 0:
            return np.array([]), np.array([])
        k = min(k, len(X_subset))
        selected = [0]
        for _ in range(1, k):
            dist = pairwise_distances(X_subset[selected], X_subset)
            min_dist = np.min(dist, axis=0)
            next_idx = np.argmax(min_dist)
            selected.append(next_idx)
        return X_subset[selected], selected

    # Select diverse in PCA space
    high_PCA, _ = farthest_point_sampling(high_candidates, n_high)
    low_PCA, _ = farthest_point_sampling(low_candidates, n_low)

    # Inverse-transform to original space
    high_original = pca.inverse_transform(high_PCA) if len(high_PCA) else np.empty((0, pca.n_features_))
    low_original = pca.inverse_transform(low_PCA) if len(low_PCA) else np.empty((0, pca.n_features_))

    return high_PCA, low_PCA, high_original, low_original



def select_diverse_data_indices(X_reduced, Z_prob, n_select=10, threshold_high=0.9, threshold_low=0.1):
    """
    Select diverse sample indices from high and low probability regions in PCA space.

    Parameters:
    - X_reduced: np.ndarray, shape (n_samples, 2), PCA-reduced representation of X
    - Z_prob: array-like, shape (n_samples,), predicted probabilities
    - n_select: int, number of diverse points to select from high and low prob regions
    - threshold_high: float, probability threshold for high region
    - threshold_low: float, probability threshold for low region

    Returns:
    - high_indices: indices in the original dataset from high probability region
    - low_indices: indices in the original dataset from low probability region
    """

    def farthest_point_sampling(X_subset, candidate_indices, k):
        selected = []
        if len(candidate_indices) == 0:
            return selected
        selected.append(candidate_indices[0])
        for _ in range(1, min(k, len(candidate_indices))):
            dists = pairwise_distances(X_subset[candidate_indices], X_subset[selected])
            min_dist = dists.min(axis=1)
            next_idx = candidate_indices[np.argmax(min_dist)]
            selected.append(next_idx)
        return selected

    # Boolean masks for high and low prob regions
    high_mask = Z_prob >= threshold_high
    low_mask = Z_prob <= threshold_low

    high_candidates = np.where(high_mask)[0]
    low_candidates = np.where(low_mask)[0]

    # Run diversity sampling based on PCA space
    high_indices = farthest_point_sampling(X_reduced, high_candidates, n_select)
    low_indices = farthest_point_sampling(X_reduced, low_candidates, n_select)

    return high_indices, low_indices



def run_diverse_selection_workflow(X, y_pred, model, df_reference, n_select=10, threshold_high=0.9, threshold_low=0.1,
                                   title="Diverse Sample Selection in PCA Space",
                                   figsize=(8, 6), cmap='viridis', plot_title_str=True):
    """
    Wrapper to run PCA, predict probabilities, select diverse samples, and plot.

    Parameters:
    - X: feature array
    - y_pred: predicted labels or placeholder
    - model: classifier with `predict_proba`
    - df_reference: original dataframe to extract selected samples
    - n_select: number of high/low diverse samples to select
    - title: plot title
    - figsize: size of figure
    - cmap: colormap for probabilities

    Returns:
    - high_samples, low_samples: DataFrames of selected samples
    - high_idxs, low_idxs: corresponding indices
    """

    # Run PCA and prediction
    _, _, Z_prob, X_reduced = pca_model_contour_plot(X=X, y_pred=y_pred, model=model, cmap=cmap, plot_title=plot_title_str)

    # Select diverse indices
    high_idxs, low_idxs = select_diverse_data_indices(X_reduced, Z_prob, n_select=n_select, threshold_high=threshold_high, threshold_low=threshold_low)

    # Extract samples
    high_samples = df_reference.iloc[high_idxs]
    low_samples = df_reference.iloc[low_idxs]

    # Plot
    plt.figure(figsize=figsize)
    scatter_plot = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=Z_prob, cmap=cmap, s=20, label='All points')
    plt.scatter(X_reduced[high_idxs, 0], X_reduced[high_idxs, 1], c='red', marker='^', s=80, label='High Diverse')
    plt.scatter(X_reduced[low_idxs, 0], X_reduced[low_idxs, 1], c='blue', marker='s', s=80, label='Low Diverse')
    plt.colorbar(scatter_plot, label='Predicted Probability')
    plt.legend()
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return high_samples, low_samples, high_idxs, low_idxs


def extract_original_concentrations(sample_df, df_info):
    """
    Given a sample dataframe and df_info containing SMILES → Name mapping,
    rename the conc columns using replacements when possible, keep original names otherwise,
    then return only the inverse-log-transformed concentration columns.
    """
    # Original component names for smi1–smi7
    original_names = [
        'Decanoic acid', 'Decanoate', 'Decylamine',
        'Decyltrimethyl ammonium bromid', 'Decyl sodium sulfate',
        'Decanol', 'Glycerol monodecanoate'
    ]

    # Get mapping from column index to new molecule (if replaced)
    replacement_mapping = {}
    for i in range(1, 8):
        smi_col = f'smi{i}'
        unique_smis = sample_df[smi_col].unique()
        for smile in unique_smis:
            matched = df_info[df_info['SMILES'] == smile]
            if not matched.empty:
                replacement_mapping[i] = matched.iloc[0]['Name']
                break

    # Create column renaming dictionary using either replacement or original names
    column_rename_dict = {}
    for i, original_name in enumerate(original_names, start=1):
        final_name = replacement_mapping.get(i, original_name)
        column_rename_dict[f'smi{i}'] = final_name
        column_rename_dict[f'conc{i}'] = final_name

    # Apply renaming
    renamed_df = sample_df.rename(columns=column_rename_dict)

    # Keep only concentration columns (every second column starting from index 1)
    conc_only_df = renamed_df.iloc[:, 1::2].copy()

    # Inverse log1p to get back the original concentrations
    original_conc_df = conc_only_df.apply(np.expm1)

    return original_conc_df

def save_objects_joblib(filepath, **kwargs):
    """
    Save multiple Python objects using joblib.

    Parameters:
    - filepath: str — path to save the .pkl file.
    - **kwargs: any number of named Python objects to save.
    """
    joblib.dump(kwargs, filepath)


def get_dispense_volume(concentrations_df, unique_solutes, allow_zero=False):

    stocks = [[50, 10, 2], [50, 10, 2], [50, 10, 2], [50, 10, 2], [50, 10, 2], [15, 3], [10, 2]]
    valid_dispense_volumes = [[(4, 20), (4, 20), (0, 20)], [(4, 20), (4, 20), (0, 20)], [(4, 20), (4, 20), (0, 20)],
                            [(4, 20), (4, 20), (0, 20)], [(4, 20), (4, 20), (0, 20)], [(4, 20), (0, 20)], [(4, 20), (0, 20)]]

    # Total volume
    total_volume = 200

    # Initialize empty DataFrame to store dispense volumes
    column_names = []
    for i, (col, stock) in enumerate(zip(concentrations_df.columns, stocks), start=1):
        for j, concentration in enumerate(stock, start=1):
            column_names.append(f'{col} ({concentration} mM)')

    dispense_df = pd.DataFrame(columns=column_names)

    # Iterate through each row of concentrations dataframe
    for index, row in concentrations_df.iterrows():
        dispense_volumes = []
        
        # Calculate required volume for each stock solution
        for i, (conc, stock, valid_vol) in enumerate(zip(row, stocks, valid_dispense_volumes), start=1):
            total_mass = conc * total_volume
            volumes = [total_mass / s for s in stock]
            
            # Initialize a list to store volumes for this stock solution
            stock_dispense_volumes = []
            
            # Flag to track if a valid volume has been encountered
            valid_volume_found = False
            
            # Check all possible volumes within the valid dispense volume range
            for volume, vol_range in zip(volumes, valid_vol):
                if not valid_volume_found and vol_range[0] <= volume <= vol_range[1]:
                    stock_dispense_volumes.append(volume)
                    # Set the flag to True to indicate a valid volume has been found
                    valid_volume_found = True
                else:
                    stock_dispense_volumes.append(0)
            
            # Append the volumes for this stock solution to dispense_volumes
            dispense_volumes.extend(stock_dispense_volumes)
            
        # Append dispense volumes for this row to the dataframe
        dispense_df.loc[index] = dispense_volumes
        # Round all numbers in the DataFrame to one decimal place
        dispense_df = dispense_df.round(1)
        # Apply custom function to each element of the DataFrame
        # dispense_df = dispense_df.applymap(discretize)

    # Iterate over the rows and remove those where all columns for the same solute are zero
    if allow_zero:
        return dispense_df
    else:
        rows_to_remove = []
        for index, row in dispense_df.iterrows():
            all_zero = True
            for solute in unique_solutes:
                solute_cols = [col for col in dispense_df.columns if solute in col]
                if all(row[col] == 0 for col in solute_cols):
                    all_zero = all_zero and True
                    rows_to_remove.append(index)
                    # print(index)
                else:
                    all_zero = False

        # Remove the identified rows
        dispense_df = dispense_df.drop(rows_to_remove)
        return dispense_df

def concentration_space_plot(df):
    conc_cols = [f'conc{i}' for i in range(1, 8)]
    X = df[conc_cols]

    # Define prediction group
    threshold = 0.5  # or adjust to 0.7 for stricter cutoff
    df['group'] = df['pred_prob'].apply(lambda x: 'Positive' if x >= threshold else 'Negative')

    # PCA reduction to 3D
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
    df_pca['group'] = df['group']
    df_pca['pred_prob'] = df['pred_prob']

    # Interactive 3D scatter plot
    fig = px.scatter_3d(
        df_pca,
        x='PC1', y='PC2', z='PC3',
        color='pred_prob',  # categorical coloring
        hover_data={'pred_prob': True},
        title='3D PCA of Concentration Space (colored by Prediction Group)',
        color_continuous_scale='RdBu_r'
    )

    fig.update_traces(marker=dict(size=5, opacity=0.8))
    fig.update_layout(
        width=800,
        height=600,
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3'
        )
    )
    fig.show()


def prepare_dispense_df_model(model, x_smi, df_new, df_info, prob_range=(0.7, 0.9), sample_n=10, random_state=None):
    """
    Select samples whose predicted probability lies within a given range,
    randomly choose N from them, transform back to concentrations,
    and generate a dispense file.

    Parameters
    ----------
    model : sklearn-like classifier with predict_proba
        Your trained classification model.
    x_smi : array-like
        Latent space representation of samples.
    df_new : DataFrame
        The original dataframe corresponding to x_smi.
    df_info : DataFrame
        Amphiphile information dataframe.
    prob_range : tuple(float, float)
        Probability range (min_prob, max_prob) for filtering.
    sample_n : int
        Number of samples to select after filtering.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dispense_df : DataFrame
        Dispense file ready for experiments.
    """
    # Predict probabilities
    y_pred = model.predict_proba(x_smi)[:, 1]

    # Filter by probability range
    mask = (y_pred >= prob_range[0]) & (y_pred <= prob_range[1])
    filtered_df = df_new[mask]
    filtered_probs = y_pred[mask]

    # Print how many samples match the range
    print(f"Found {len(filtered_df)} samples in range {prob_range}")

    if len(filtered_df) == 0:
        raise ValueError(f"No samples found in probability range {prob_range}")

    # Randomly select N samples
    selected_df = filtered_df.sample(
        n=min(sample_n, len(filtered_df)),
        random_state=random_state
    )

    # Transform back to concentrations
    original_concentrations = extract_original_concentrations(selected_df, df_info)

    # Get dispense file
    solutes = original_concentrations.columns
    dispense_df = get_dispense_volume(original_concentrations, solutes, allow_zero=True)

    return original_concentrations, dispense_df, selected_df, filtered_probs

def get_feature_color(feature, color_map):
    '''
    amphiphile_colors = {
    # First set (7 amphiphiles)
    'Decanoic acid': 'blue',
    'Decanoate': 'orange',
    'Decylamine': 'green',
    'Decyltrimethyl ammonium bromid': 'red',
    'Decyl sodium sulfate': 'purple',
    'Decanol': 'brown',
    'Glycerol monodecanoate': 'pink',

    # Second set (8 amphiphiles)
    'Decanal': 'cyan',               # bright cyan
    'Geraniol': 'lime',              # neon-like lime
    'Hexadecanoic acid': 'gold',     # strong yellow-gold
    'Myristoleic acid': 'deepskyblue', # vivid blue (different from "blue")
    'Glycine octylester': 'crimson', # strong red-pink (not same as red/pink)
    'Perfluorooctanoic acid': 'darkviolet', # deep violet
    'Tridecafluorooctane-1-sulphonic acid': 'black', # solid black
    'Heptadecafluorooctanesulfonic acid': 'olive'    # earthy olive green
    }
    '''
    # Sort keys by length (longest first) to avoid substring conflicts
    for key in sorted(color_map.keys(), key=len, reverse=True):
        if key.lower() in feature.lower():
            return color_map[key]
    return "gray"  # default if no match

def plot_samples_composition(df, title):
    num_samples = df.shape[0]
    features = df.columns

    plt.figure(figsize=(15, num_samples * 0.5))

    used_labels = set()  # to avoid duplicate legend entries

    for i in range(num_samples):
        sample = df.iloc[i, :]
        left = 0
        for feature in features:
            # print(feature)
            color = get_feature_color(feature)
            # print(color)
            # print('='*10)
            label = None
            if feature not in used_labels:
                label = feature
                used_labels.add(feature)
            plt.barh(
                i,
                sample[feature],
                left=left,
                color=color,
                label=label
            )
            left += sample[feature]

    plt.xlabel('Concentration (mM)')
    plt.ylabel('Sample Index')
    plt.title(title)
    # Put the legend on the bottom
    plt.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.1), 
        ncol=4, 
        frameon=False
    )
    plt.tight_layout()
    plt.show()
