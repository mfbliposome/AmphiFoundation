import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def expand_dataset(df, n_samples=10, cmc_range=(1e-6, 10.0)):
    """
    Expand dataset by keeping the original row and generating n_samples CMC values in a given range.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe containing at least 'CMC', 'pCMC', 'w'
    n_samples : int
        Number of generated samples per row (in addition to the original)
    cmc_range : tuple
        Range (min, max) for sampling new CMC values

    Returns
    -------
    pd.DataFrame
        Expanded dataframe with label column
    """
    expanded_rows = []

    for _, row in df.iterrows():
        cmc_true = row["CMC"]

        # keep original row
        orig_row = row.copy()
        orig_row["label"] = 1  # treat original as ">= itself"
        expanded_rows.append(orig_row)

        # generate n_samples random CMC values
        # generated_cmc = np.random.uniform(cmc_range[0], cmc_range[1], n_samples)# not random, use linspace?
        # generated_cmc = np.linspace(cmc_range[0], cmc_range[1], n_samples)
        
        # 5 values below (exclude cmc_true itself)
        cmc_below = np.linspace(cmc_range[0], cmc_true, 6)[:-1]
        
        # 5 values above (exclude cmc_true itself)
        cmc_above = np.linspace(cmc_true, cmc_range[1], 6)[1:]
        
        generated_cmc = np.concatenate([cmc_below, cmc_above])

        for new_cmc in generated_cmc:
            new_row = row.copy()
            new_row["CMC"] = new_cmc
            new_row["pCMC"] = -np.log10(new_cmc)
            new_row["w"] = np.log1p(new_cmc)
            new_row["label"] = 0 if new_cmc < cmc_true else 1
            expanded_rows.append(new_row)

    expanded_df = pd.DataFrame(expanded_rows)
    return expanded_df

def expand_dataset_pCMC(df, n_samples=10, pcmc_range=(0.1, 10.0)):
    """
    Expand dataset by keeping the original row and generating n_samples values in pCMC space.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe containing at least 'CMC', 'pCMC', 'w'
    n_samples : int
        Number of generated samples per row (in addition to the original)
    pcmc_range : tuple
        Range (min, max) for sampling new pCMC values

    Returns
    -------
    pd.DataFrame
        Expanded dataframe with label column
    """
    expanded_rows = []

    for _, row in df.iterrows():
        pcmc_true = row["pCMC"]
        cmc_true = row["CMC"]

        # keep original row
        orig_row = row.copy()
        orig_row["label"] = 1
        expanded_rows.append(orig_row)

        # generate 5 values below and 5 values above in pCMC space
        pcmc_below = np.linspace(pcmc_range[0], pcmc_true, 6)[:-1]
        pcmc_above = np.linspace(pcmc_true, pcmc_range[1], 6)[1:]
        generated_pcmc = np.concatenate([pcmc_below, pcmc_above])

        for new_pcmc in generated_pcmc:
            new_cmc = 10 ** (-new_pcmc)  # convert back to CMC
            new_row = row.copy()
            new_row["CMC"] = new_cmc
            new_row["pCMC"] = new_pcmc
            new_row["w"] = np.log1p(new_cmc)
            new_row["label"] = 1 if new_pcmc < pcmc_true else 0
            expanded_rows.append(new_row)

    expanded_df = pd.DataFrame(expanded_rows)
    return expanded_df



def make_weighted_features(df):
    # get all feature columns (everything before 'w')
    feature_cols = df.columns[:df.columns.get_loc("w")]
    
    # multiply features by w
    features_weighted = df[feature_cols].multiply(df["w"], axis=0)
    
    # add label column
    features_weighted["label"] = df["label"].values
    
    return features_weighted


def prepare_dataset_clf(df, n_samples=10, cmc_range=(0.1, 10.0), expand_cri='pCMC'):
    """
    Full workflow:
    1. Compute CMC from pCMC
    2. Compute w = ln(1+CMC), reorder columns
    3. Expand dataset (keep original + n_samples generated CMC values)
    4. Build weighted feature dataset (w * features + label)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with latent features + Surfactant_Type + pCMC
    n_samples : int
        Number of generated CMC values per row
    cmc_range : tuple
        Range (min, max) for generating CMC values

    Returns
    -------
    df_expanded : pd.DataFrame
        Dataset with original + generated CMC values, including w and label
    df_weighted : pd.DataFrame
        Dataset with only weighted features and label
    """
    
    #  add CMC and w 
    df = df.copy()
    df["CMC"] = 10 ** (-df["pCMC"])
    df["w"] = np.log1p(df["CMC"])
    
    # reorder so w comes before Surfactant_Type
    cols = list(df.columns)
    cols.remove("w")
    insert_idx = cols.index("Surfactant_Type")
    cols = cols[:insert_idx] + ["w"] + cols[insert_idx:]
    df = df[cols]
    
    if expand_cri=='pCMC':
        df_expanded = expand_dataset_pCMC(df, n_samples, cmc_range)
    else:
        df_expanded = expand_dataset(df, n_samples, cmc_range)
    
    df_weighted = make_weighted_features(df_expanded)
    
    return df_expanded, df_weighted



def predict_pcmc_binary_search(clf, df_weighted, cmc_min=1e-6, cmc_max=10.0, tol=1e-6, plot=False):
    """
    Predict pCMC using binary search with classifier clf.
    Works only on the original rows in df_weighted (every 11th row).
    
    Parameters
    ----------
    clf : trained classifier
    df_weighted : DataFrame (with original+generated rows)
    cmc_min : float, lower bound for search
    cmc_max : float, upper bound for search
    tol : float, tolerance for stopping criterion
    plot : bool, whether to plot true vs predicted
    
    Returns
    -------
    result_df : DataFrame with true_cmc, pred_cmc, true_pCMC, pred_pCMC
    rmse : float (calculated on pCMC)
    r2 : float (calculated on pCMC)
    """

    df_orig = df_weighted.iloc[::11].copy()  # keep only original rows
    X_cols = [c for c in df_weighted.columns if c not in ["CMC", "pCMC", "label"]]

    predictions = []

    for _, row in df_orig.iterrows():
        low, high = cmc_min, cmc_max
        while abs(high - low) > tol:
            mid = (low + high) / 2
            candidate = row.copy()
            candidate["CMC"] = mid
            candidate["w"] = np.log1p(mid)

            X_mid = candidate[X_cols].values.reshape(1, -1)
            pred = clf.predict(X_mid)[0]

            if pred == 1:
                high = mid
            else:
                low = mid

        pred_cmc = (low + high) / 2
        predictions.append((row["CMC"], pred_cmc))

    result_df = pd.DataFrame(predictions, columns=["true_cmc", "pred_cmc"])
    result_df["true_pCMC"] = -np.log10(result_df["true_cmc"])
    result_df["pred_pCMC"] = -np.log10(result_df["pred_cmc"])

    # Metrics in pCMC space
    rmse = np.sqrt(mean_squared_error(result_df["true_pCMC"], result_df["pred_pCMC"]))
    r2 = r2_score(result_df["true_pCMC"], result_df["pred_pCMC"])

    if plot:
        plt.figure(figsize=(6, 6))
        plt.scatter(result_df["true_pCMC"], result_df["pred_pCMC"], alpha=0.7, edgecolor="k")
        plt.plot([result_df["true_pCMC"].min(), result_df["true_pCMC"].max()],
                 [result_df["true_pCMC"].min(), result_df["true_pCMC"].max()],
                 "r--", label="Ideal")
        plt.xlabel("True pCMC")
        plt.ylabel("Predicted pCMC")
        plt.title(f"True vs Predicted pCMC\nRMSE={rmse:.3f}, R²={r2:.3f}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return result_df, rmse, r2

def predict_pcmc_binary_search_pCMC(clf, df_ori, cmc_min=0.1, cmc_max=10.0, tol=1e-2, plot=False):
    """
    Predict pCMC using binary search with classifier clf.
    Works only on the original rows in df_weighted (every 11th row).
    
    Parameters
    ----------
    clf : trained classifier
    df_weighted : DataFrame (with original+generated rows)
    cmc_min : float, lower bound for search
    cmc_max : float, upper bound for search
    tol : float, tolerance for stopping criterion
    plot : bool, whether to plot true vs predicted
    
    Returns
    -------
    result_df : DataFrame with true_cmc, pred_cmc, true_pCMC, pred_pCMC
    rmse : float (calculated on pCMC)
    r2 : float (calculated on pCMC)
    """
    df_orig = df_ori.iloc[::11].copy()
    # May just use this line
    X_cols = [c for c in df_ori.columns if c not in ["Surfactant_Type", "pCMC"]]
    # X_cols = [c for c in df_weighted.columns if c not in ["CMC", "pCMC", "label"]]
    # print(X_cols)

    predictions = []

    for _, row in df_orig.iterrows():
        low, high = cmc_min, cmc_max
        while abs(high - low) > tol:
            mid = (low + high) / 2
            # Need according this to get output from clf
            candidate = row.copy()
            candidate["pCMC"] = mid
            candidate["CMC"] = 10 ** (-mid)
            candidate["w"] = np.log1p(candidate["CMC"])

            # X_mid = candidate[X_cols].values.reshape(1, -1)
            X_mid = (candidate[X_cols] * candidate["w"]).values.reshape(1, -1)
            pred = clf.predict(X_mid)[0]
            # print(pred)

            if pred == 1:
                low = mid
            else:
                high = mid
        #     print(high)
        #     print(mid)
        #     print(low)
        #     print('--'*5)
        # print('=='*10)

        pred_cmc = (low + high) / 2
        predictions.append((row["pCMC"], pred_cmc))

    result_df = pd.DataFrame(predictions, columns=["true_pCMC", "pred_pCMC"])

    # Metrics in pCMC space
    rmse = np.sqrt(mean_squared_error(result_df["true_pCMC"], result_df["pred_pCMC"]))
    r2 = r2_score(result_df["true_pCMC"], result_df["pred_pCMC"])

    if plot:
        plt.figure(figsize=(6, 6))
        plt.scatter(result_df["true_pCMC"], result_df["pred_pCMC"], alpha=0.7, edgecolor="k")
        plt.plot([result_df["true_pCMC"].min(), result_df["true_pCMC"].max()],
                 [result_df["true_pCMC"].min(), result_df["true_pCMC"].max()],
                 "r--", label="Ideal")
        plt.xlabel("True pCMC")
        plt.ylabel("Predicted pCMC")
        plt.title(f"True vs Predicted pCMC\nRMSE={rmse:.3f}, R²={r2:.3f}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return result_df, rmse, r2
