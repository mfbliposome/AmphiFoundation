from datetime import datetime
import torch
import numpy as np
import pickle
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

from src.utils import get_latent_space, run_classifier, run_classifier_update
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import auc


def get_sample_fromGP(model_path, info_path, n_samples = 1000, feature='conc7', save_file=False, random_seed=42):
    
    model_gp = joblib.load(model_path)

    df_info = pd.read_csv(info_path)

    # Sample from GPs
    np.random.seed(random_seed)
    # bounds in log1p space (shape: 2 x 7)
    bounds = torch.tensor([[0., 0., 0., 0., 0., 0., 0.],
                        [5., 5., 5., 5., 5., 1.5, 1.0]])
    bounds = torch.log1p(bounds)

    # Sample n_samples points uniformly within bounds
    dim = bounds.shape[1]
    low, high = bounds[0].numpy(), bounds[1].numpy()

    X_sampled = np.random.uniform(low=low, high=high, size=(n_samples, dim))

    # Predict class probabilities or labels
    # probabilities: probs = model.predict_proba(X_sampled)
    y_sampled = model_gp.predict(X_sampled)

    # Combine into final dataset
    columns = [f'feature_{i+1}' for i in range(dim)]
    df_sampled = pd.DataFrame(X_sampled, columns=columns)
    df_sampled['label'] = y_sampled

    df = df_sampled.copy()
    # Create mapping from df_sampled column names to df_info["Name"]
    column_to_name = {
        'feature_1': 'Decanoic acid',
        'feature_2': 'Decanoate',
        'feature_3': 'Decylamine',
        'feature_4': 'Decyltrimethyl ammonium bromid',
        'feature_5': 'Decyl sodium sulfate',
        'feature_6': 'Decanol',
        'feature_7': 'Glycerol monodecanoate'
    }

    # Get SMILES mapping from df_info
    name_to_smiles = dict(zip(df_info['Name'], df_info['SMILES']))

    # Construct new dataframe
    new_data = {}

    for col in df.columns[:-1]:  # Skip num_vesicles
        chem_name = column_to_name[col]
        smiles = name_to_smiles.get(chem_name, '')
        new_data[f"{col}_SMILES"] = [smiles] * len(df)
        new_data[f"{col}_Concentration"] = df[col]

    # Add num_vesicles
    new_data['num_vesicles'] = df['label']

    df_structured = pd.DataFrame(new_data)

    # Rename columns to the desired format
    new_column_names = [
        'smi1', 'conc1',
        'smi2', 'conc2',
        'smi3', 'conc3',
        'smi4', 'conc4',
        'smi5', 'conc5',
        'smi6', 'conc6',
        'smi7', 'conc7',
        'vesicles_formation'
    ]

    # Apply the new column names
    df_structured.columns = new_column_names

    df1 = df_structured 

    if save_file == True:
        # Save results dir
        time_str = datetime.now().strftime('%Y%m%d_%H')
        save_dir = f'../../results/{feature}_{time_str}'
        os.makedirs(save_dir, exist_ok=True)
        df1.to_csv(os.path.join(save_dir, f'df1.csv'), index=False)

    return df1, model_gp

def get_sample_lack_feature(df, model, feature='conc7'):
    df_deficit = df.copy()
    df_deficit[feature] = 0
    # Now based on GP model and the features after removing, get the new predicted labels

    # Extract concentration features
    conc_columns = ['conc1', 'conc2', 'conc3', 'conc4', 'conc5', 'conc6', 'conc7']
    X_new = df_deficit[conc_columns]

    # Predict using the model
    new_labels = model.predict(X_new)

    df_deficit_newlabel = df_deficit.copy()
    # Replace the old 'vesicles_formation' column
    df_deficit_newlabel['vesicles_formation'] = new_labels

    df2 = df_deficit_newlabel

    # Save results dir
    time_str = datetime.now().strftime('%Y%m%d_%H')
    save_dir = f'../results/{feature}_{time_str}'
    os.makedirs(save_dir, exist_ok=True)
    df2.to_csv(os.path.join(save_dir, f'df2_{feature}.csv'), index=False)

    return df2

def get_sample_lack_features_mul(df, model, feature_list_to_zero):
    """
    Set selected concentration features to 0 and get new predictions.

    Parameters:
    - df: Original dataframe
    - model: Trained model with a predict method
    - feature_list_to_zero: List of feature names to be set to zero (e.g., ['conc1', 'conc3'])

    Returns:
    - DataFrame with updated 'vesicles_formation' predictions
    """
    df_deficit = df.copy()

    # Check if feature_list_to_zero is in correct form
    if not isinstance(feature_list_to_zero, list) or not all(isinstance(feat, str) for feat in feature_list_to_zero):
        raise ValueError("`feature_list_to_zero` must be a list of strings, e.g., ['conc5', 'conc6'].")

    # Set selected features to 0
    for feature in feature_list_to_zero:
        if feature in df_deficit.columns:
            df_deficit[feature] = 0

    # Predict with updated input
    conc_columns = ['conc1', 'conc2', 'conc3', 'conc4', 'conc5', 'conc6', 'conc7']
    X_new = df_deficit[conc_columns]
    new_labels = model.predict(X_new)

    # Create new dataframe with updated labels
    df_deficit['vesicles_formation'] = new_labels

    # Save results dir
    time_str = datetime.now().strftime('%Y%m%d_%H')
    save_dir = f'../results/{feature_list_to_zero}_{time_str}'
    os.makedirs(save_dir, exist_ok=True)
    df_deficit.to_csv(os.path.join(save_dir, f'df2_{feature_list_to_zero}.csv'), index=False)

    return df_deficit

def classifier_performance(df_full, df_deficit, feature='conc7'):
    # Get the latent space of removed feature dataset
    x_smi = get_latent_space(df_deficit)
    # Get the latent space of full feature dataset
    x_smi_full = get_latent_space(df_full)

    xtrain = x_smi
    ytrain = df_deficit['vesicles_formation']
    xpred = x_smi_full
    model_clf, y_pred, roc_auc, fpr, tpr, threshold= run_classifier(xtrain, ytrain, xpred)

    # Save results dir
    time_str = datetime.now().strftime('%Y%m%d_%H')
    save_dir = f'../../results/{feature}_{time_str}'
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots()
    ax.set_title("ROC-AUC Curve")
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC ({feature}_removed)')
    ax.legend(loc='lower right')

    # Save plot
    plot_path = os.path.join(save_dir, f'ROC_{feature}_{time_str}.png')
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Save ROC Curve Data 
    df_roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': threshold})
    df_roc.to_csv(os.path.join(save_dir, f'ROC_data_{feature}_{time_str}.csv'), index=False)

    # Save y_pred
    df_pred = pd.DataFrame({
        'y_true': ytrain.values,
        'y_pred_prob': y_pred
    })
    df_pred.to_csv(os.path.join(save_dir, f'y_pred_{feature}_{time_str}.csv'), index=False)

    # Classification Report & Confusion Matrix 
    y_pred_binary = (y_pred >= 0.5).astype(int)
    report = classification_report(ytrain, y_pred_binary, output_dict=True)
    cm_ori = confusion_matrix(ytrain, y_pred_binary)
    if cm_ori.shape == (1, 1):
        # Only one class present in ytrain, fill the matrix to (2, 2) with zeros
        cm_full = np.zeros((2, 2), dtype=int)
        present_class = ytrain.iloc[0]  # could be 0 or 1
        cm_full[present_class, present_class] = cm_ori[0, 0]
        cm = cm_full
    else:
        cm = cm_ori

    # Save classification report
    pd.DataFrame(report).transpose().to_csv(
        os.path.join(save_dir, f'classification_report_{feature}_{time_str}.csv')
    )

    # Save confusion matrix
    pd.DataFrame(cm, index=['True_0', 'True_1'], columns=['Pred_0', 'Pred_1']).to_csv(
        os.path.join(save_dir, f'confusion_matrix_{feature}_{time_str}.csv')
    )

    return model_clf, y_pred, roc_auc, fpr, tpr, threshold

def classifier_performance_update(df_full, df_deficit, feature='conc7', classifier_alter=False):
    # Get the latent space of removed feature dataset
    x_smi = get_latent_space(df_deficit)
    # Get the latent space of full feature dataset
    x_smi_full = get_latent_space(df_full)

    xtrain = x_smi
    ytrain = df_deficit['vesicles_formation']
    xtest = x_smi_full
    ytest = df_full['vesicles_formation']
    if not classifier_alter:
        model_clf, y_pred, roc_auc, fpr, tpr, threshold= run_classifier_update(xtrain, ytrain, xtest, ytest)
    else:
        model_clf, y_pred, roc_auc, fpr, tpr, threshold= run_classifier_update(xtrain, ytrain, xtest, ytest, classifier_alter=True )


    # Save results dir
    time_str = datetime.now().strftime('%Y%m%d_%H')
    save_dir = f'../results/{feature}_{time_str}'
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots()
    ax.set_title("ROC-AUC Curve")
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC ({feature}_removed)')
    ax.legend(loc='lower right')

    # Save plot
    plot_path = os.path.join(save_dir, f'ROC_{feature}_{time_str}.png')
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Save ROC Curve Data
    df_roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': threshold})
    df_roc.to_csv(os.path.join(save_dir, f'ROC_data_{feature}_{time_str}.csv'), index=False)

    #Save y_pred
    df_pred = pd.DataFrame({
        'y_true': ytest.values,
        'y_pred_prob': y_pred
    })
    df_pred.to_csv(os.path.join(save_dir, f'y_pred_{feature}_{time_str}.csv'), index=False)

    # Classification Report & Confusion Matrix 
    y_pred_binary = (y_pred >= 0.5).astype(int)
    report = classification_report(ytest, y_pred_binary, output_dict=True)
    cm_ori = confusion_matrix(ytest, y_pred_binary)
    if cm_ori.shape == (1, 1):
        # Only one class present in ytrain, fill the matrix to (2, 2) with zeros
        cm_full = np.zeros((2, 2), dtype=int)
        present_class = ytest.iloc[0]  # could be 0 or 1
        cm_full[present_class, present_class] = cm_ori[0, 0]
        cm = cm_full
    else:
        cm = cm_ori

    # Save classification report
    pd.DataFrame(report).transpose().to_csv(
        os.path.join(save_dir, f'classification_report_{feature}_{time_str}.csv')
    )

    # Save confusion matrix
    pd.DataFrame(cm, index=['True_0', 'True_1'], columns=['Pred_0', 'Pred_1']).to_csv(
        os.path.join(save_dir, f'confusion_matrix_{feature}_{time_str}.csv')
    )

    return model_clf, y_pred, roc_auc, fpr, tpr, threshold


def run_multiple(feature_str, model_path, info_path, n_runs=10):
    fpr_list, tpr_list, auc_list = [], [], []

    for i in range(n_runs):
        df1, model_gp = get_sample_fromGP(model_path, info_path, n_samples=1000, feature = feature_str, save_file=False, random_seed=i)
        df2 = get_sample_lack_feature(df1, model_gp, feature=feature_str)
        model_clf, y_pred, roc_auc, fpr, tpr, threshold = classifier_performance_update(df1, df2, feature=feature_str)

        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(roc_auc)

    # Define common FPR grid for interpolation
    mean_fpr = np.linspace(0, 1, 100)

    # Interpolate all TPRs
    tprs_interp = []
    for fpr, tpr in zip(fpr_list, tpr_list):
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
    tprs_interp = np.array(tprs_interp)

    # Mean and std
    mean_tpr = np.mean(tprs_interp, axis=0)
    std_tpr = np.std(tprs_interp, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(auc_list)

    # Plot 
    plt.figure(figsize=(7,6))

    # Plot all individual ROC curves
    for fpr, tpr in zip(fpr_list, tpr_list):
        plt.plot(fpr, tpr, color="gray", alpha=0.4, lw=0.8)

    # Plot mean ROC curve
    plt.plot(mean_fpr, mean_tpr, color="royalblue",
             label=f"Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})", lw=2)

    # Shaded region for ± std
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color="blue", alpha=0.2)

    # Chance line
    plt.plot([0, 1], [0, 1], linestyle="--", color="orange")

    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title(f"ROC Curves Across {n_runs} Runs (Feature removed: {feature_str})", fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"../../results/roc_curves_{feature_str}.png", dpi=300, bbox_inches="tight")
    plt.show()

def run_multiple_multi_feature(feature_str, model_path, info_path, n_runs=10):
    fpr_list, tpr_list, auc_list = [], [], []

    for i in range(n_runs):
        df1, model_gp = get_sample_fromGP(model_path, info_path, n_samples=1000, feature = feature_str, save_file=False, random_seed=i)
        df2 = get_sample_lack_features_mul(df1, model_gp, feature_list_to_zero=feature_str)
        model_clf, y_pred, roc_auc, fpr, tpr, threshold = classifier_performance_update(df1, df2, feature=feature_str)

        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(roc_auc)

    # Define common FPR grid for interpolation
    mean_fpr = np.linspace(0, 1, 100)

    # Interpolate all TPRs
    tprs_interp = []
    for fpr, tpr in zip(fpr_list, tpr_list):
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
    tprs_interp = np.array(tprs_interp)

    # Mean and std
    mean_tpr = np.mean(tprs_interp, axis=0)
    std_tpr = np.std(tprs_interp, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(auc_list)

    # Plot
    plt.figure(figsize=(7,6))

    # Plot all individual ROC curves
    for fpr, tpr in zip(fpr_list, tpr_list):
        plt.plot(fpr, tpr, color="gray", alpha=0.4, lw=0.8)

    # Plot mean ROC curve
    plt.plot(mean_fpr, mean_tpr, color="royalblue",
             label=f"Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})", lw=2)

    # Shaded region for ± std
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color="blue", alpha=0.2)

    # Chance line
    plt.plot([0, 1], [0, 1], linestyle="--", color="orange")

    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title(f"Feature removed: {feature_str}", fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"../../results/roc_curves_{feature_str}.png", dpi=600, bbox_inches="tight")
    plt.show()

    
