
import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from models.smi_ted.smi_ted_light.load import load_smi_ted
from utils import binarize_last_column

def get_data_AL(file_path, info_path):
    '''
    This function is to get the input data set
    Data set comes from the active learning paper
    '''
    df_input_update_ori = pd.read_csv(file_path) 
    df_input = df_input_update_ori.copy()
    df_input.iloc[:,0:7] = df_input.iloc[:,0:7].applymap(lambda x: np.log1p(x))
    df_input=df_input.iloc[:, 0:8]
    print(df_input.shape)

    df_info = pd.read_csv(info_path)
    df = df_input.copy()
    df.columns = [col.replace('_Concentration (mM)', '') for col in df.columns[:-1]] + ['num_vesicles']

    column_to_name = {
        'decanoic acid': 'Decanoic acid',
        'decanoate': 'Decanoate',
        'decylamine': 'Decylamine',
        'decyl trimethylamine': 'Decyltrimethyl ammonium bromid',
        'decylsulfate': 'Decyl sodium sulfate',
        'decanol': 'Decanol',
        'monocaprin': 'Glycerol monodecanoate'
    }

    # Get SMILES mapping from df_info
    name_to_smiles = dict(zip(df_info['Name'], df_info['SMILES']))

    new_data = {}

    for col in df.columns[:-1]:  # Skip num_vesicles
        chem_name = column_to_name[col]
        smiles = name_to_smiles.get(chem_name, '')
        new_data[f"{col}_SMILES"] = [smiles] * len(df)
        new_data[f"{col}_Concentration"] = df[col]

    new_data['num_vesicles'] = df['num_vesicles']

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

    df_structured.columns = new_column_names
    print(df_structured.shape)

    return df_structured

def get_latent_space_c(df):
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

    return df_train_emb, x_smi


def perturb_and_embed(df, row_indices, component_idx, num_points=5, custom_conc_series=None):
    """
    Perturb concentration of one component in multiple rows and get latent space representations.
    
    Parameters:
    - df: pd.DataFrame — original df_structured
    - row_indices: list of integers — row indices to perturb
    - component_idx: integer (1-7) — which component to perturb (smi1 ~ smi7)
    - num_points: int — number of points (used only if custom_conc_series not provided)
    - custom_conc_series: array/list (optional) — user-defined concentration values (in log1p space)
    
    Returns:
    - results: dict {row_idx: {'conc_values': [...], 'x_smi_list': [...]}}  
    - df_new_all: dataframe for compositoins
    """
    conc_col = f'conc{component_idx}'
    results = {}
    df_new_all = {}
    
    # Determine concentration series to use
    if custom_conc_series is None:
        min_conc = df[conc_col].min()
        max_conc = df[conc_col].max()
        new_conc_series = np.linspace(min_conc, max_conc, num_points)
    else:
        new_conc_series = np.array(custom_conc_series)

    for row_idx in row_indices:
        x_smi_list = []
        df_list = []
        
        for conc in new_conc_series:
            df_new = df.copy()
            df_new.at[row_idx, conc_col] = conc
            df_list.append(df_new.copy())
            
            # Compute latent space 
            df_train_emb, x_smi = get_latent_space_c(df_new.iloc[[row_idx]])
            x_smi_list.append(x_smi)
        
        results[row_idx] = {
            'conc_values': new_conc_series,
            'x_smi_list': x_smi_list
        }
        df_new_all[row_idx] = df_list
    
    return results, df_new_all

def replace_entire_smi_column(df, smi_column, new_smi):
    """
    Replace the entire smi_column in df with a single new SMILES.

    Parameters:
    - df: pd.DataFrame — df_structured
    - smi_column: str — column to replace (e.g., "smi3")
    - new_smi: str — SMILES string to set for the whole column

    Returns:
    - df_new: pd.DataFrame 
    """
    df_new = df.copy()

    if smi_column not in df.columns:
        raise ValueError(f"{smi_column} not found in DataFrame columns.")
    
    df_new[smi_column] = new_smi

    return df_new

def get_plot(x_smi_list, model_clf, conc_values):

    y_pred = np.empty(len(x_smi_list))

    for i in range(0, len(x_smi_list)):
        y_pred[i] = model_clf.predict_proba(x_smi_list[i])[:, 1]

    plt.plot(conc_values, y_pred, marker='o')  
    plt.xlabel('Concentration Values')
    plt.ylabel('Predicted Probability')
    plt.title('Predicted Probability vs. Concentration Values')
    plt.grid(True)
    plt.show()


def perturb_predict_plot(df, row_indices, component_idx, model_path, 
                         num_points=5, custom_conc_series=None):
    """
    For each row index, perturb concentration, predict probability, and plot curve.

    Parameters:
    - df: pd.DataFrame — df_structured
    - row_indices: list of integers — rows to perturb
    - component_idx: integer (1-7) — component to perturb (smi1 ~ smi7)
    - model_path: str — path to trained classifier pickle file
    - num_points: int — number of points in conc series (used if custom_conc_series not given)
    - custom_conc_series: list/array (optional) — custom conc series

    Returns:
    - results: dict
    """
    # Load model
    with open(model_path, "rb") as f:
        model_clf = pickle.load(f)
    
    # Perturb + embed
    results = perturb_and_embed(
        df=df, 
        row_indices=row_indices, 
        component_idx=component_idx, 
        num_points=num_points, 
        custom_conc_series=custom_conc_series
    )
    
    # Plot for each row
    for row_idx in row_indices:
        conc_values = results[row_idx]['conc_values']
        x_smi_list = results[row_idx]['x_smi_list']
        
        y_pred = np.empty(len(x_smi_list))
        for i in range(len(x_smi_list)):
            y_pred[i] = model_clf.predict_proba(x_smi_list[i])[:, 1]
        
        plt.plot(conc_values, y_pred, marker='o')
        plt.xlabel('Concentration Values')
        plt.ylabel('Predicted Probability')
        plt.title(f'Sample {row_idx}')
        plt.grid(True)
        plt.show()
    
    return results

def perturb_predict_plot_subplots(df, component_idx, model_path, 
                                   num_points=5, custom_conc_series=None, 
                                   n_samples=7, selected_indices=None, step=48, figsize=(18, 8)):
    """
    Systematically select rows (every `step`), perturb concentration, predict, and plot subplots.

    Parameters:
    - df: pd.DataFrame — df_structured
    - component_idx: integer (1-7) — component to perturb (smi1 ~ smi7)
    - model_path: str — path to trained classifier pickle file
    - num_points: int — number of points in conc series (used if custom_conc_series not given)
    - custom_conc_series: list/array (optional) — custom conc series
    - n_samples: int — number of samples to plot (default 7)
    - step: int — interval step between samples (default 48)
    - figsize: tuple — size of the entire figure

    Returns:
    - results: dict 
    - selected_indices: list of row indices used
    """
    # Load model
    with open(model_path, "rb") as f:
        model_clf = pickle.load(f)
    
    # Default option: systematically select indices: 0, 48, 96, ...
    if selected_indices is None:
        selected_indices = [i * step for i in range(n_samples)]
    
    # Perturb + embed
    results = perturb_and_embed(
        df=df, 
        row_indices=selected_indices, 
        component_idx=component_idx, 
        num_points=num_points, 
        custom_conc_series=custom_conc_series
    )
    
    # Plot subplots
    ncols = n_samples
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    if n_samples == 1:
        axes = [axes]  

    for idx, row_idx in enumerate(selected_indices):
        conc_values = results[row_idx]['conc_values']
        x_smi_list = results[row_idx]['x_smi_list']
        
        y_pred = np.empty(len(x_smi_list))
        for i in range(len(x_smi_list)):
            y_pred[i] = model_clf.predict_proba(x_smi_list[i])[:, 1]
        
        ax = axes[idx]
        ax.plot(conc_values, y_pred, marker='o')
        ax.set_title(f'Sample {row_idx}')
        ax.grid(True)
        if idx == 0:
            ax.set_ylabel('Predicted Prob.')
        ax.set_xlabel('Conc.')

    plt.tight_layout()
    plt.show()
    
    return results, selected_indices



def perturb_predict_compare_plot_subplots(df, component_idx, model_clf, model_gp, 
                                          num_points=5, custom_conc_series=None, 
                                          n_samples=7, selected_indices=None, step=48, figsize=(18, 8)):
    """
    Systematically select rows, perturb concentration, predict with two models, and plot comparison subplots.

    Parameters:
    - df: pd.DataFrame — df_structured
    - component_idx: integer (1-7) — component to perturb (smi1 ~ smi7)
    - model_clf_path: str — path to trained classifier pickle file 
    - model_gp: trained classifier (takes conc1 ~ conc7 as input)
    - num_points: int — number of points in conc series (used if custom_conc_series not given)
    - custom_conc_series: list/array (optional) — custom conc series
    - n_samples: int — number of samples to plot (default 7)
    - selected_indices: list (optional) — custom row indices to use
    - step: int — interval step between samples (default 48)
    - figsize: tuple — size of the entire figure

    Returns:
    - results: dict
    - selected_indices: list of row indices used
    - df_new_all: dataframe for compositoins 
    """
    
    # Default option: systematically select indices: 0, 48, 96, ...
    if selected_indices is None:
        selected_indices = [i * step for i in range(n_samples)]
    
    # Perturb + embed
    results, df_new_all  = perturb_and_embed(
        df=df, 
        row_indices=selected_indices, 
        component_idx=component_idx, 
        num_points=num_points, 
        custom_conc_series=custom_conc_series
    )
    
    # Plot subplots
    ncols = n_samples
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    if n_samples == 1:
        axes = [axes] 

    # Concentration feature columns
    conc_cols = [f'conc{i}' for i in range(1, 8)]

    for idx, row_idx in enumerate(selected_indices):
        conc_values = results[row_idx]['conc_values']
        x_smi_list = results[row_idx]['x_smi_list']
        
        # Predict using model_clf 
        y_pred_clf = np.empty(len(x_smi_list))
        for i in range(len(x_smi_list)):
            y_pred_clf[i] = model_clf.predict_proba(x_smi_list[i])[:, 1]
        
        # Predict using model_gp 
        y_pred_gp = []
        for conc in conc_values:
            df_perturbed_row = df.copy()
            df_perturbed_row.at[row_idx, f'conc{component_idx}'] = conc
            conc_features = df_perturbed_row.loc[row_idx, conc_cols].values.reshape(1, -1)
            prob_gp = model_gp.predict_proba(conc_features)[:, 1]
            y_pred_gp.append(prob_gp[0])
        y_pred_gp = np.array(y_pred_gp)
        
        # Plot both
        ax = axes[idx]
        ax.plot(conc_values, y_pred_clf, marker='o', color='blue', label='Model_FM')
        ax.plot(conc_values, y_pred_gp, marker='s', linestyle='--', color='k', label='Model_GP')
        ax.set_title(f'Sample {row_idx}')
        ax.grid(True)
        if idx == 0:
            ax.set_ylabel('Predicted Probabilities')
        ax.set_xlabel('Concentration')
        ax.legend(fontsize=8)
        # Save individual subplot as its own figure 
        fig_indiv, ax_indiv = plt.subplots(figsize=(6, 4))
        ax_indiv.plot(conc_values, y_pred_clf, marker='o', color='blue', label='Model_FM')
        ax_indiv.plot(conc_values, y_pred_gp, marker='s', linestyle='--', color='k', label='Model_GP')
        # ax_indiv.set_title(f'Sample {row_idx}')
        ax_indiv.set_ylabel('Predicted Probabilities')
        ax_indiv.set_xlabel('Concentration')
        # ax_indiv.grid(True)
        # ax_indiv.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"../results/sample_{row_idx}_fmvsgp.png", dpi=600)
        plt.close(fig_indiv)

    plt.tight_layout()
    plt.savefig("../../results/all_samples_fmvsgp.png", dpi=600)
    plt.show()
    
    return results, selected_indices, df_new_all

def compare_results_multiple_runs(results_list, model_clf, selected_indices, custom_conc_series,
                                   labels=None, colors=None, figsize=(21, 4), legend_loc='bottom'):
    """
    Compare predicted probabilities from multiple results dicts on same samples with shared legend outside.

    Parameters:
    - results_list: list of results dicts [results0, results1, ...]
    - model_clf: trained classifier (model_clf_ran)
    - selected_indices: list of row indices (same across all results dicts)
    - custom_conc_series: list/array of conc values used (same across all runs)
    - labels: list of str (optional) — labels for each result set (default = Run 1, Run 2,...)
    - colors: list of colors (optional) — plot colors (default = ['blue', 'red', 'green', ...])
    - figsize: tuple — size of the figure
    - legend_loc: str — 'bottom' or 'right'

    Returns:
    - None (plots directly)
    """
    n_runs = len(results_list)
    n_samples = len(selected_indices)

    if labels is None:
        labels = [f'Run {i+1}' for i in range(n_runs)]
    if colors is None:
        default_colors = ['blue', 'red', 'green', 'orange', 'purple']
        colors = default_colors[:n_runs]

    ncols = n_samples
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    if n_samples == 1:
        axes = [axes]  

    legend_handles = []
    legend_labels = []

    for idx, row_idx in enumerate(selected_indices):
        ax = axes[idx]

        for run_idx, results in enumerate(results_list):
            conc_values = results[row_idx]['conc_values']
            x_smi_list = results[row_idx]['x_smi_list']
            
            # Predict prob for all x_smi
            y_pred = np.empty(len(x_smi_list))
            for i in range(len(x_smi_list)):
                y_pred[i] = model_clf.predict_proba(x_smi_list[i])[:, 1]

            line, = ax.plot(conc_values, y_pred, marker='o', color=colors[run_idx], label=labels[run_idx])
            
            # Store handle/label only once (from first subplot)
            if idx == 0:
                legend_handles.append(line)
                legend_labels.append(labels[run_idx])

        ax.set_title(f'Sample {row_idx}')
        ax.grid(True)
        if idx == 0:
            ax.set_ylabel('Predicted Probabilities')
        ax.set_xlabel('Concentrations')

        # Save individual figure
        fig_indiv, ax_indiv = plt.subplots(figsize=(6, 4))
        for run_idx, results in enumerate(results_list):
            conc_values = results[row_idx]['conc_values']
            x_smi_list = results[row_idx]['x_smi_list']
            y_pred = np.empty(len(x_smi_list))
            for i in range(len(x_smi_list)):
                y_pred[i] = model_clf.predict_proba(x_smi_list[i])[:, 1]
            ax_indiv.plot(conc_values, y_pred, marker='o', color=colors[run_idx], label=labels[run_idx])
        # ax_indiv.set_title(f'Sample {row_idx}')
        ax_indiv.set_ylabel("Predicted Probabilities")
        ax_indiv.set_xlabel("Concentrations")
        # ax_indiv.legend(fontsize=8)
        # ax_indiv.grid(True)
        fig_indiv.tight_layout()
        fig_indiv.savefig(f"../results/sample_{row_idx}_individual_concsweep.png", dpi=600, bbox_inches='tight')
        plt.close(fig_indiv)

    # Adjust layout and add shared legend outside
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    if legend_loc == 'bottom':
        fig.legend(legend_handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, 0.0),
                   ncol=n_runs, fontsize=10, frameon=False)
    elif legend_loc == 'right':
        fig.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1.01, 0.5),
                   fontsize=10, frameon=False)

    fig.savefig("../../results/all_samples_conc_sweep.png", dpi=600)
    plt.show()
