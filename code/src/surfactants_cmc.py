from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from models.smi_ted.smi_ted_light.load import load_smi_ted
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def get_train_data(df_data):
    train_smiles_list = df_data['SMILES']
    model_SMI = load_smi_ted(folder='../src/models/smi_ted/smi_ted_light', ckpt_filename='smi-ted-Light_40.pt')

    return_tensor=True
    with torch.no_grad():
        x_emb = model_SMI.encode(train_smiles_list, return_torch=return_tensor)

    x_emb_array  = x_emb.numpy()
    x_emb_frame = pd.DataFrame(x_emb_array)

    df_total = pd.concat([x_emb_frame, df_data.iloc[:,1:]], axis=1)

    return x_emb_frame, df_total

def train_model(data, property='pCMC'):
    time_str = datetime.now().strftime('%Y%m%d_%H')
    save_dir = f'../../results/{property}_{time_str}'
    os.makedirs(save_dir, exist_ok=True)
    # Split the dataframe
    train_df, test_df = train_test_split(
        data,
        test_size=0.2, 
        stratify=data['Surfactant_Type'],
        random_state=42  # for reproducibility
    )

    print("Train size:", train_df.shape)
    print("Test size:", test_df.shape)
    print(train_df['Surfactant_Type'].value_counts())
    print(test_df['Surfactant_Type'].value_counts())

    train_x = train_df.iloc[:,0:768]
    train_y = train_df[property]

    test_x = test_df.iloc[:,0:768]
    test_y = test_df[property]

    regressor = RandomForestRegressor(random_state=42)
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
    plt.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'r--', lw=2) 
    plt.xlabel(f'True {property}', fontsize=14)
    plt.ylabel(f'Predicted {property}', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, f'prediction_{property}_{time_str}.png')
    plt.savefig(plot_path, dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

    # Merge test_x, test_y, pred_y, and Surfactant_Type for plotting
    test_df = test_df.copy()
    test_df['True'] = test_y
    test_df['Predicted'] = pred_y

    # Color mapping
    surfactant_types = test_df['Surfactant_Type'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(surfactant_types)))  
    color_dict = dict(zip(surfactant_types, colors))

    plt.figure(figsize=(7, 7))

    # Plot each surfactant type separately
    for surfactant in surfactant_types:
        subset = test_df[test_df['Surfactant_Type'] == surfactant]
        plt.scatter(
            subset['True'], subset['Predicted'],
            label=surfactant,
            color=color_dict[surfactant],
            edgecolor='k',
            alpha=0.7
        )

    plt.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'r--', lw=2)

    plt.xlabel(f'True {property}', fontsize=14)
    plt.ylabel(f'Predicted {property}', fontsize=14)
    plt.legend(title='Surfactant Type', loc='best', frameon=True)

    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(save_dir, f'prediction_colored_{property}_{time_str}.png')
    plt.savefig(plot_path, dpi=300)
    plt.show()
    plt.close()

    return model, RMSE_score, r2, train_df, test_df

