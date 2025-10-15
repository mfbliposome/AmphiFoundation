from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
from xgboost import XGBClassifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from typing import Tuple
from typing import Optional

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import os

def run_classifier_model(xtrain, ytrain, xtest, ytest, random_seed, classifier_alter=False ):
    if classifier_alter == False:
        # Train XGBoost classifier
        clf = XGBClassifier(n_estimators=5000, learning_rate=0.01, max_depth=10, use_label_encoder=False, eval_metric='logloss')
        clf.fit(xtrain, ytrain)

        # Predict probabilities for ROC-AUC
        # y_pred = xgb_clf.predict(xtest)
        y_pred = clf.predict_proba(xtest)[:,1]

    elif classifier_alter == True:
        clf = RandomForestClassifier(
            n_estimators=500,  
            max_depth=10,      
            random_state=random_seed,
            
        )
        clf.fit(xtrain, ytrain)
        # class_weight='balanced'  # Helps if data is imbalanced
        # Predict probabilities for ROC-AUC
        y_pred = clf.predict_proba(xtest)[:, 1]

    # ROC-AUC score
    roc_auc = roc_auc_score(ytest, y_pred)
    fpr, tpr, threshold = roc_curve(ytest, y_pred)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    return clf, y_pred, roc_auc, fpr, tpr, threshold


def evaluate_model_performance(
    df: pd.DataFrame,
    label_col: str,
    output_csv_name: str = "roc_auc_results",
    n_runs: int = 10,
    test_size: float = 0.2,
    classifier_alter: bool = True,
    save_plot_name: str = 'roc_auc_curve',
    data_type: str = None,
    model_type: str = None,
    random_state: int = 42
):
    """
    Evaluate classifier performance over multiple random splits and plot ROC-AUC curves.

    Parameters:
    - df: Input DataFrame.
    - label_col: Name of the column for classification labels.
    - output_csv: File path to save ROC-AUC scores.
    - n_runs: Number of train/test split runs.
    - test_size: Fraction of test size.
    - classifier_alter: Passed to `run_classifier_model`.
    - save_plot_name: plot name.
    - data_type: "surfactants", "binary _system", or "amphiphiles".
    - model_type: 'Baseline', 'VICGAE', 'Chemprop', 'Chemeleon', 'SMI-TED'
    - random_state: Base seed to allow reproducibility.
    """
    all_fpr = np.linspace(0, 1, 100)
    roc_aucs = []
    interp_tprs = []

    rng = np.random.default_rng(seed=random_state)
    all_classifiers = []

    for i in range(n_runs):
        random_states = rng.integers(0, 10000)
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_states,
            stratify=df[label_col] # use balanced data
        )
        train_x = train_df.drop(columns=[label_col])
        train_y = train_df[label_col]
        test_x = test_df.drop(columns=[label_col])
        test_y = test_df[label_col]

        # Run classifier
        clf, y_pred, roc_auc, fpr, tpr, threshold = run_classifier_model(
            train_x, train_y, test_x, test_y, random_seed = random_states, classifier_alter=classifier_alter
        )
        all_classifiers.append(clf)

        # Interpolate TPRs to standard FPR grid
        # Reason to do this: can’t just average TPRs directly unless they share the same FPR points.
        interp_tpr = np.interp(all_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)
        roc_aucs.append(roc_auc)

    # Convert to arrays
    interp_tprs = np.array(interp_tprs)
    mean_tpr = interp_tprs.mean(axis=0)
    std_tpr = interp_tprs.std(axis=0)

    # Save AUC scores
    auc_df = pd.DataFrame({
        'run': list(range(n_runs)),
        'roc_auc': roc_aucs
    })
    mean_auc = np.mean(roc_aucs)
    std_auc = np.std(roc_aucs)

    auc_df['mean_auc'] = mean_auc
    auc_df['std_auc'] = std_auc

    auc_df.to_csv(f"../../results/{model_type}_{data_type}_{output_csv_name}.csv", index=False)

    # Plot
    fig, ax = plt.subplots()
    plot_title = f"{model_type}_{data_type}"
    ax.set_title(plot_title)
    for i, tpr in enumerate(interp_tprs):
        ax.plot(all_fpr, tpr, lw=0.8, alpha=0.4, color='gray' )

    # ax.plot(all_fpr, mean_tpr, color='darkorange', lw=2, label=f"Mean ROC (AUC = {np.mean(roc_aucs):.4f})")
    ax.plot(all_fpr, mean_tpr, color='#1f77b4', lw=2, label=f"Mean ROC (AUC = {np.mean(roc_aucs):.4f})")

    ax.fill_between(all_fpr, mean_tpr - 2*std_tpr, mean_tpr + 2*std_tpr, color='#aec7e8', alpha=0.2, label='±2 std dev')
    ax.plot([0, 1], [0, 1], color='#ff7f0e', lw=1, linestyle='--')
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')

    if save_plot_name:
        plt.savefig(f"../../results/{model_type}_{data_type}_{save_plot_name}.png", dpi=600, bbox_inches='tight')
    plt.show()

    return auc_df, all_classifiers

# Usage
# evaluate_model_performance(
#     df=df_example,
#     label_col="vesicles_formation",
#     n_runs=10,
#     data_type='amphiphiles',
#     model_type='VICGAE'
# )


def evaluate_model_performance_regression(
    df: pd.DataFrame,
    feature_cols: Optional[list] = None,
    target_col: str = "pCMC",
    stratify_col: Optional[str] = "Surfactant_Type",
    n_runs: int = 10,
    data_type: str = None,
    model_type: str = None,
    save_csv_name: str = "regression_performance",
    save_plot_name: str = "qq_plot",
    random_state: int = 42
) -> pd.DataFrame:
    
    # Default feature columns: all columns except last 2
    if feature_cols is None:
        feature_cols = df.columns[:-2].tolist()

    rmse_list = []
    r2_list = []

    last_test_y = None
    last_pred_y = None
    rng = np.random.default_rng(seed=random_state)
    
    for i in range(n_runs):
        random_states = rng.integers(0, 10000)
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            stratify=df[stratify_col] if stratify_col else None,
            random_state=random_states,
        )

        train_x = train_df[feature_cols]
        train_y = train_df[target_col]
        test_x = test_df[feature_cols]
        test_y = test_df[target_col]

        regressor = RandomForestRegressor(n_estimators=100, random_state=random_states)
        model = TransformedTargetRegressor(
            regressor=regressor,
            transformer=MinMaxScaler(feature_range=(-1, 1))
        ).fit(train_x, train_y)

        pred_y = model.predict(test_x)
        rmse = np.sqrt(mean_squared_error(test_y, pred_y))
        r2 = r2_score(test_y, pred_y)

        rmse_list.append(rmse)
        r2_list.append(r2)

        # Save the last run for scatter plot
        last_test_y = test_y
        last_pred_y = pred_y

        print(f"[Run {i+1}] RMSE: {rmse:.4f}, R²: {r2:.4f}")

    # Create DataFrame to store results
    result_df = pd.DataFrame({
        "run": list(range(1, n_runs + 1)),
        "RMSE": rmse_list,
        "R2": r2_list
    })

    result_df.to_csv(f"../../results/{model_type}_{data_type}_{save_csv_name}.csv", index=False)
    # print(f"Saved results to {save_csv_name}")

    # Boxplots with mean and std annotated
    plot_title = f"{model_type}_{data_type}"
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].boxplot(rmse_list, patch_artist=True, boxprops=dict(facecolor='#1f77b4', color='#1f77b4'),
            medianprops=dict(color='white'),
            whiskerprops=dict(color='#1f77b4'),
            capprops=dict(color='#1f77b4'),
            flierprops=dict(markerfacecolor='#1f77b4', markeredgecolor='#1f77b4'))
    ax[0].set_title(f"{model_type}_{data_type}_RMSE Distribution")
    ax[0].set_ylabel("RMSE")
    mean_rmse = np.mean(rmse_list)
    std_rmse = np.std(rmse_list)
    ax[0].text(1.1, mean_rmse, f"Mean={mean_rmse:.3f}\nStd={std_rmse:.3f}", va="center")

    ax[1].boxplot(r2_list, patch_artist=True, boxprops=dict(facecolor='#1f77b4', color='#1f77b4'),
            medianprops=dict(color='white'),
            whiskerprops=dict(color='#1f77b4'),
            capprops=dict(color='#1f77b4'),
            flierprops=dict(markerfacecolor='#1f77b4', markeredgecolor='#1f77b4'))
    ax[1].set_title( f"{model_type}_{data_type}_R² Distribution")
    ax[1].set_ylabel("R²")
    mean_r2 = np.mean(r2_list)
    std_r2 = np.std(r2_list)
    ax[1].text(1.1, mean_r2, f"Mean={mean_r2:.3f}\nStd={std_r2:.3f}", va="center")
    plt.tight_layout()
    save_path = f"../../results/{model_type}_{data_type}_regression_metrics.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()

    # Scatter plot from last run
    plt.figure()
    plt.scatter(test_y, pred_y, color='#1f77b4', alpha=0.6, edgecolor='#1f77b4')  
    plt.plot([test_y.min(), test_y.max()], 
             [test_y.min(), test_y.max()], 
             color='#ff7f0e', linestyle='--', lw=1)  # Dark gray line

    plt.xlabel(f'True {target_col}', fontsize=14)
    plt.ylabel(f'Predicted {target_col}', fontsize=14)
    plt.title(plot_title)
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../../results/{model_type}_{data_type}_{save_plot_name}.png", dpi=600)
    # print(f"Saved scatter plot to {save_scatter_path}")
    plt.show()

    return result_df
# Usage
# results_df = evaluate_model_performance_regression(df_example, data_type='surfactants', model_type='VICGAE')

def evaluate_model_performance_regression_no_normalization(
    df: pd.DataFrame,
    feature_cols: Optional[list] = None,
    target_col: str = "pCMC",
    stratify_col: Optional[str] = "Surfactant_Type",
    n_runs: int = 10,
    random_seed: int = 42,
    data_type: str = None,
    model_type: str = None,
    save_csv_name: str = "regression_performance",
    save_plot_name: str = "qq_plot",
    random_state: int = 42
) -> pd.DataFrame:
    # Default feature columns: all columns except last 2
    if feature_cols is None:
        feature_cols = df.columns[:-2].tolist()

    rmse_list = []
    r2_list = []

    last_test_y = None
    last_pred_y = None
    rng = np.random.default_rng(seed=random_state)
    
    for i in range(n_runs):
        random_states = rng.integers(0, 10000)
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            stratify=df[stratify_col] if stratify_col else None,
            random_state=random_states,
        )

        train_x = train_df[feature_cols]
        train_y = train_df[target_col]
        test_x = test_df[feature_cols]
        test_y = test_df[target_col]

        regressor = RandomForestRegressor(n_estimators=100, random_state=random_states)
        model = regressor.fit(train_x, train_y)

        pred_y = model.predict(test_x)
        rmse = np.sqrt(mean_squared_error(test_y, pred_y))
        r2 = r2_score(test_y, pred_y)

        rmse_list.append(rmse)
        r2_list.append(r2)

        # Save the last run for scatter plot
        last_test_y = test_y
        last_pred_y = pred_y

        print(f"[Run {i+1}] RMSE: {rmse:.4f}, R²: {r2:.4f}")

    # Create DataFrame to store results
    result_df = pd.DataFrame({
        "run": list(range(1, n_runs + 1)),
        "RMSE": rmse_list,
        "R2": r2_list
    })

    result_df.to_csv(f"../../results/{model_type}_{data_type}_{save_csv_name}.csv", index=False)
    # print(f"Saved results to {save_csv_name}")

    # Boxplots with mean and std annotated
    plot_title = f"{model_type}_{data_type}"
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].boxplot(rmse_list, patch_artist=True, boxprops=dict(facecolor='#1f77b4', color='#1f77b4'),
            medianprops=dict(color='white'),
            whiskerprops=dict(color='#1f77b4'),
            capprops=dict(color='#1f77b4'),
            flierprops=dict(markerfacecolor='#1f77b4', markeredgecolor='#1f77b4'))
    ax[0].set_title(f"{model_type}_{data_type}_RMSE Distribution")
    ax[0].set_ylabel("RMSE")
    mean_rmse = np.mean(rmse_list)
    std_rmse = np.std(rmse_list)
    ax[0].text(1.1, mean_rmse, f"Mean={mean_rmse:.3f}\nStd={std_rmse:.3f}", va="center")

    ax[1].boxplot(r2_list, patch_artist=True, boxprops=dict(facecolor='#1f77b4', color='#1f77b4'),
            medianprops=dict(color='white'),
            whiskerprops=dict(color='#1f77b4'),
            capprops=dict(color='#1f77b4'),
            flierprops=dict(markerfacecolor='#1f77b4', markeredgecolor='#1f77b4'))
    ax[1].set_title( f"{model_type}_{data_type}_R² Distribution")
    ax[1].set_ylabel("R²")
    mean_r2 = np.mean(r2_list)
    std_r2 = np.std(r2_list)
    ax[1].text(1.1, mean_r2, f"Mean={mean_r2:.3f}\nStd={std_r2:.3f}", va="center")
    plt.tight_layout()
    save_path = f"../../results/{model_type}_{data_type}_regression_metrics.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()

    # Scatter plot from last run
    plt.figure()
    plt.scatter(test_y, pred_y, color='#1f77b4', alpha=0.6, edgecolor='#1f77b4')  
    plt.plot([test_y.min(), test_y.max()], 
             [test_y.min(), test_y.max()], 
             color='#ff7f0e', linestyle='--', lw=1)  # Dark gray line

    plt.xlabel(f'True {target_col}', fontsize=14)
    plt.ylabel(f'Predicted {target_col}', fontsize=14)
    plt.title(plot_title)
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../../results/{model_type}_{data_type}_{save_plot_name}.png", dpi=600)
    # print(f"Saved scatter plot to {save_scatter_path}")
    plt.show()

    return result_df


def visualize_latent_space(
        df_total: pd.DataFrame, 
        method_list=['pca', 'tsne', 'umap'], 
        supervised_umap=True, 
        data_type: str = None,
        model_type: str = None):
    
    # Determine type based on columns
    if 'Surfactant_Type' in df_total.columns and 'pCMC' in df_total.columns:
        # Type 1: Regression with surfactant category
        X = df_total.iloc[:, :-2].values
        y = df_total['pCMC'].values
        label_type = 'regression'
        d_type = 'regression'
    elif 'miscibility' in df_total.columns:
        # Type 2: Classification
        X = df_total.iloc[:, :-1].values
        y = df_total['miscibility'].values
        label_type = 'binary_classification'
        d_type = 'binary_classification'
    elif 'vesicles_formation' in df_total.columns:
        # Type 3: Binary classification
        X = df_total.iloc[:, :-1].values
        y = df_total['vesicles_formation'].values
        label_type = 'amphiphile_classification'
        d_type = 'amphiphile_classification'
    else:
        raise ValueError("Unrecognized df_total format. Please ensure it includes appropriate target columns.")
    target_col = df_total.columns[-1]
    color_by_category = False

    # Handle color: classification (0/1), categorical, or regression
    if df_total[target_col].dtype.kind in 'fi' and len(np.unique(y)) > 10:
        label_type = 'regression'
        color_values = plt.cm.Blues((y - np.min(y)) / (np.max(y) - np.min(y)))
    else:
        label_type = 'classification'
        # if set(np.unique(y)).issubset({0, 1}):
        #     color_values = ["#d33c0e" if val == 1 else '#1f77b4' for val in y]  # blue/gray
        if set(np.unique(y)).issubset({0, 1}):
            color_values = ["#FF6B6B" if val == 1 else "#4D9DE0" for val in y]  # brighter red and blue

        else:
            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(y)
            # color_values = plt.get_cmap("tab10")(y_encoded)
            palette = sns.color_palette("Set2", n_colors=len(np.unique(y_encoded)))  # bright/pastel colors
            color_values = np.array(palette)[y_encoded]

    # Optional: if 'Surfactant_Type' is present, use it for coloring
    if 'Surfactant_Type' in df_total.columns:
        surfactant_labels = df_total['Surfactant_Type']
        encoder = LabelEncoder()
        # color_values = plt.get_cmap("tab10")(encoder.fit_transform(surfactant_labels))
        palette = sns.color_palette("Set2", n_colors=len(np.unique(surfactant_labels)))
        color_values = np.array(palette)[encoder.fit_transform(surfactant_labels)]
        color_by_category = True

    def plot_2d(X_emb, model_type, data_type, method_name):
        fig, ax = plt.subplots(figsize=(6, 5))
        title = f"{model_type}_{data_type}_{method_name}"
        ax.set_title(f"{title} (2D)", fontsize=13)

        if label_type == 'regression' and 'Surfactant_Type' in df_total.columns:
            unique_labels = np.unique(surfactant_labels)
            # cmap = plt.get_cmap("tab20")
            # cmap = plt.get_cmap("Set2")
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
            color_dict = dict(zip(unique_labels, colors))

            for i, label in enumerate(unique_labels):
                mask = surfactant_labels == label
                ax.scatter(X_emb[mask, 0], X_emb[mask, 1],
                        color=color_dict[label], edgecolor='k', alpha=0.7, label=label)

            # Place legend outside the plot
            # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, borderaxespad=0.)
            # legend_fig, legend_ax = plt.subplots(figsize=(2, len(unique_labels)*0.3))
            legend_fig, legend_ax = plt.subplots(figsize=(len(unique_labels)*1.2, 1.5))
            legend_ax.axis('off')

            # Create handles and labels from color_dict
            handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                                markerfacecolor=color_dict[label], markersize=8) 
                    for label in unique_labels]

            legend_ax.legend(handles=handles, loc='center', fontsize=9, frameon=False,
                                            ncol=len(unique_labels), handletextpad=0.5)  
            legend_fig.tight_layout()          
            legend_fig.savefig(f"../../results/legend_{model_type}_{data_type}_{method_name}.png", dpi=600, bbox_inches='tight')
            plt.close(legend_fig)
        elif d_type == 'binary_classification':
            # color_values = ["#FF6B6B" if val == 1 else "#4D9DE0" for val in y]
            ax.scatter(X_emb[:, 0], X_emb[:, 1], c=color_values, edgecolor='k', alpha=0.7)

            unique_labels = np.unique(y)
            if len(unique_labels) <= 20:  # avoid massive legends
                # legend_fig, legend_ax = plt.subplots(figsize=(2, len(unique_labels)*0.3))
                legend_fig, legend_ax = plt.subplots(figsize=(len(unique_labels)*1.2, 1.5))  # wider for horizontal
                legend_ax.axis('off')

                label_color_map = {}
                for col, lab in zip(color_values, y):
                    if lab not in label_color_map:
                        label_color_map[lab] = col

                label_map = {0: "immiscible", 1: "miscible"} 
                handles = []
                for label in unique_labels:
                    color = label_color_map.get(label)
                    # if color is a numpy array, convert to tuple (matplotlib accepts tuples)
                    if isinstance(color, np.ndarray):
                        color = tuple(color)
                    # fallback color if something unexpected happens
                    if color is None:
                        color = plt.cm.tab10[int(label) % 10] if isinstance(label, (int, np.integer)) else "#333333"

                    handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                            label=label_map.get(label, str(label)),
                                            markerfacecolor=color, markersize=8))

                legend_ax.legend(handles=handles, loc='center', fontsize=9, frameon=False,
                                 ncol=len(unique_labels), handletextpad=0.5)
                legend_fig.tight_layout()
                legend_fig.savefig(f"../../results/legend_{model_type}_{data_type}_{method_name}.png",
                                dpi=600, bbox_inches='tight')
                plt.close(legend_fig)

        elif d_type == 'amphiphile_classification':
            ax.scatter(X_emb[:, 0], X_emb[:, 1], c=color_values, edgecolor='k', alpha=0.7)

            unique_labels = np.unique(y)
            if len(unique_labels) <= 20:  # avoid massive legends
                # legend_fig, legend_ax = plt.subplots(figsize=(2, len(unique_labels)*0.3))
                legend_fig, legend_ax = plt.subplots(figsize=(len(unique_labels)*1.2, 1.5))  # wider for horizontal
                legend_ax.axis('off')

                label_color_map = {}
                for col, lab in zip(color_values, y):
                    if lab not in label_color_map:
                        label_color_map[lab] = col

                label_map = {0: "No Vesicle Formation", 1: "Vesicle Formation"}
                handles = []
                for label in unique_labels:
                    color = label_color_map.get(label)
                    # if color is a numpy array, convert to tuple (matplotlib accepts tuples)
                    if isinstance(color, np.ndarray):
                        color = tuple(color)
                    # fallback color if something unexpected happens
                    if color is None:
                        color = plt.cm.tab10[int(label) % 10] if isinstance(label, (int, np.integer)) else "#333333"

                    handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                            label=label_map.get(label, str(label)),
                                            markerfacecolor=color, markersize=8))

                legend_ax.legend(handles=handles, loc='center', fontsize=9, frameon=False,
                                ncol=len(unique_labels), handletextpad=0.5)
                legend_fig.tight_layout()
                legend_fig.savefig(f"../../results/legend_{model_type}_{data_type}_{method_name}.png",
                                dpi=600, bbox_inches='tight')
                plt.close(legend_fig)

        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        fig.tight_layout()
        fig.subplots_adjust(right=0.75)  # Add space for the legend
        plt.savefig(f"../../results/latent_vis_{model_type}_{data_type}_{method_name}_2D.png", dpi=600, bbox_inches='tight')
        plt.close()



    def plot_3d(X_emb, model_type, data_type, method_name):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        title = f"{model_type}_{data_type}_{method_name}"
        ax.set_title(f"{title} (3D)")

        if label_type == 'regression' and 'Surfactant_Type' in df_total.columns:
            unique_labels = np.unique(surfactant_labels)
            # cmap = plt.get_cmap("tab20")
            # cmap = plt.get_cmap("Set2")
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
            color_dict = dict(zip(unique_labels, colors))
            for i, label in enumerate(unique_labels):
                mask = surfactant_labels == label
                ax.scatter(X_emb[mask, 0], X_emb[mask, 1], X_emb[mask, 2],
                        color=[color_dict[label]], edgecolor='k', alpha=0.7, label=label)

            # Move legend outside for 3D
            # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, borderaxespad=0.)
            
        else:
            ax.scatter(X_emb[:, 0], X_emb[:, 1], X_emb[:, 2],
                    c=color_values, edgecolor='k', alpha=0.7)

        fig.tight_layout()
        fig.subplots_adjust(right=0.75)
        plt.savefig(f"../../results/latent_vis_{model_type}_{data_type}_{method_name}_3D.png", dpi=600, bbox_inches='tight')
        plt.close()


    for method in method_list:
        if method == 'pca':
            pca_2d = PCA(n_components=2, random_state=42).fit_transform(X)
            pca_3d = PCA(n_components=3, random_state=42).fit_transform(X)
            plot_2d(pca_2d, model_type, data_type, method_name=method)
            plot_3d(pca_3d, model_type, data_type, method_name=method)

        elif method == 'tsne':
            tsne_2d = TSNE(n_components=2, random_state=42).fit_transform(X)
            tsne_3d = TSNE(n_components=3, random_state=42).fit_transform(X)
            plot_2d(tsne_2d, model_type, data_type, method_name=method)
            plot_3d(tsne_3d, model_type, data_type, method_name=method)

        elif method == 'umap':
            if supervised_umap and label_type == 'classification':
                reducer_2d = umap.UMAP(n_components=2, random_state=42).fit(X, y)
                reducer_3d = umap.UMAP(n_components=3, random_state=42).fit(X, y)
            else:
                reducer_2d = umap.UMAP(n_components=2, random_state=42).fit(X)
                reducer_3d = umap.UMAP(n_components=3, random_state=42).fit(X)

            umap_2d = reducer_2d.transform(X)
            umap_3d = reducer_3d.transform(X)
            plot_2d(umap_2d, model_type, data_type, method_name=method)
            plot_3d(umap_3d, model_type, data_type, method_name=method)
# Usage
# visualize_latent_space(df_example, data_type='surfactants',model_type='VICGAE')

def plot_results_surf_NN(results_surf_NN, save_dir="../../results"):
    os.makedirs(save_dir, exist_ok=True)

    model_names = list(results_surf_NN['rmse_results'].keys())

    for model_name in model_names:
        rmse_list = np.array(results_surf_NN['rmse_results'][model_name])
        r2_list = np.array(results_surf_NN['r2_results'][model_name])
        # y_pred = np.array(results_surf_NN['y_pred_results'][model_name])
        # y_true = np.array(results_surf_NN['y_true_results'][model_name])
        y_pred = np.array(results_surf_NN['y_pred_results'][model_name])[-1].flatten()
        y_true = np.array(results_surf_NN['y_true_results'][model_name])[-1].flatten()
        # print(y_true.shape)

        # Boxplots (RMSE and R²) 
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        color = '#1f77b4'

        # RMSE
        ax[0].boxplot(rmse_list, patch_artist=True,
                      boxprops=dict(facecolor=color, color=color),
                      medianprops=dict(color='white'),
                      whiskerprops=dict(color=color),
                      capprops=dict(color=color),
                      flierprops=dict(markerfacecolor=color, markeredgecolor=color))
        ax[0].set_title(f"{model_name}_RMSE Distribution")
        ax[0].set_ylabel("RMSE")
        mean_rmse = np.mean(rmse_list)
        std_rmse = np.std(rmse_list)
        ax[0].text(1.1, mean_rmse, f"Mean={mean_rmse:.3f}\nStd={std_rmse:.3f}", va="center")

        # R²
        ax[1].boxplot(r2_list, patch_artist=True,
                      boxprops=dict(facecolor=color, color=color),
                      medianprops=dict(color='white'),
                      whiskerprops=dict(color=color),
                      capprops=dict(color=color),
                      flierprops=dict(markerfacecolor=color, markeredgecolor=color))
        ax[1].set_title(f"{model_name}_R² Distribution")
        ax[1].set_ylabel("R²")
        mean_r2 = np.mean(r2_list)
        std_r2 = np.std(r2_list)
        ax[1].text(1.1, mean_r2, f"Mean={mean_r2:.3f}\nStd={std_r2:.3f}", va="center")

        plt.tight_layout()
        boxplot_path = os.path.join(save_dir, f"{model_name}_regression_metrics.png")
        plt.savefig(boxplot_path, dpi=600, bbox_inches='tight')
        plt.show()

        # Scatter plot 
        plt.figure()
        plt.scatter(y_true, y_pred, color=color, alpha=0.6, edgecolor=color)
        plt.plot([y_true.min(), y_true.max()],
                 [y_true.min(), y_true.max()],
                 color='#ff7f0e', linestyle='--', lw=1)
        plt.xlabel('True pCMC', fontsize=14)
        plt.ylabel('Predicted pCMC', fontsize=14)
        plt.title(f"{model_name}_True vs Predicted")
        plt.tight_layout()

        scatter_path = os.path.join(save_dir, f"{model_name}_scatter.png")
        plt.savefig(scatter_path, dpi=600, bbox_inches='tight')
        plt.show()

        print(f"Saved plots for {model_name}:\n  Boxplot: {boxplot_path}\n  Scatter: {scatter_path}\n")



def plot_roc_classification(results_dict, save_dir="../../results", data_type="binary"):
    """
    Plot ROC curves for multiple runs and compute mean ± std AUC per model.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary with keys ['y_prob_results', 'y_true_results'] where each 
        contains model names mapping to arrays of shape (n_runs, n_samples).
    save_dir : str
        save path (e.g., "../../results/NN").
    data_type : str, optional
        Label for the data type, default = "binary".
    """
    os.makedirs(save_dir, exist_ok=True)
    for model_name in results_dict['y_prob_results'].keys():
        y_probs = np.array(results_dict['y_prob_results'][model_name])  # (n_runs, n_samples)
        y_trues = np.array(results_dict['y_true_results'][model_name])  # (n_runs, n_samples)
        n_runs = y_probs.shape[0]

        all_fpr = np.linspace(0, 1, 100)
        interp_tprs = []
        roc_aucs = []

        # Compute ROC for each run
        for run in range(n_runs):
            y_true = y_trues[run]
            y_prob = y_probs[run]

            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            roc_aucs.append(roc_auc)

            interp_tpr = np.interp(all_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)

        # Convert to arrays
        interp_tprs = np.array(interp_tprs)
        mean_tpr = interp_tprs.mean(axis=0)
        std_tpr = interp_tprs.std(axis=0)

        # Save AUC statistics
        auc_df = pd.DataFrame({
            'run': list(range(n_runs)),
            'roc_auc': roc_aucs
        })
        mean_auc = np.mean(roc_aucs)
        std_auc = np.std(roc_aucs)
        auc_df['mean_auc'] = mean_auc
        auc_df['std_auc'] = std_auc

        # Save CSV
        csv_path = os.path.join(save_dir, f"{model_name}_{data_type}_roc_auc.csv")
        auc_df.to_csv(csv_path, index=False)

        # Plot ROC
        fig, ax = plt.subplots(figsize=(6, 5))
        plot_title = f"{model_name} ({data_type})"
        ax.set_title(plot_title, fontsize=13)

        # Plot all runs (gray)
        for i, tpr in enumerate(interp_tprs):
            ax.plot(all_fpr, tpr, lw=0.8, alpha=0.4, color='gray')

        # Mean ROC
        ax.plot(all_fpr, mean_tpr, color='#1f77b4', lw=2,
                label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})")

        # Std shading
        ax.fill_between(all_fpr, mean_tpr - 2*std_tpr, mean_tpr + 2*std_tpr,
                        color='#aec7e8', alpha=0.3, label='±2 std dev')

        # Diagonal
        ax.plot([0, 1], [0, 1], color='#ff7f0e', lw=1, linestyle='--')

        # Formatting
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.legend(loc='lower right', fontsize=9)

        # Save plot
        save_path = os.path.join(save_dir, f"{model_name}_{data_type}_roc_curve.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close(fig)
