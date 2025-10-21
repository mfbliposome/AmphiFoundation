import json
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import optuna

def run_NN(df, plot=False, seed=None, save_dir=None):

    # Load and Prepare Data
    # =======================
    end_column_index = df.columns.get_loc('Surfactant_Type')
    X = df.iloc[:, :end_column_index].values.astype(np.float32)  # features
    y = df["pCMC"].values.astype(np.float32).reshape(-1, 1)  # target

    # scale features
    x_scaler = StandardScaler()
    X = x_scaler.fit_transform(X)

    # scale target as well
    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # convert to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Define Model Class
    # =======================
    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dims, dropout=0.2):
            super(MLP, self).__init__()
            layers = []
            prev_dim = input_dim
            for h in hidden_dims:
                layers.append(nn.Linear(prev_dim, h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev_dim = h
            layers.append(nn.Linear(prev_dim, 1))  # regression output
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)
        
    # Training Function
    # =======================
    def train_model(model, train_loader, val_loader, epochs, lr, device):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                        factor=0.5, patience=5, verbose=True)

        model.to(device)
        best_loss = float("inf")
        patience = 20
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            model.eval()
            val_preds, val_truths = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    val_preds.append(pred.cpu().numpy())
                    val_truths.append(yb.cpu().numpy())

            val_preds = np.vstack(val_preds)
            val_truths = np.vstack(val_truths)

            # RMSE on validation
            val_rmse = np.sqrt(mean_squared_error(val_truths, val_preds))

            # Scheduler step
            scheduler.step(val_rmse)

            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val RMSE: {val_rmse:.4f}")

            # Early stopping 
            if val_rmse < best_loss:
                best_loss = val_rmse
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Final evaluation on validation set
        model.eval()
        with torch.no_grad():
            preds, truths = [], []
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds.append(model(xb).cpu().numpy())
                truths.append(yb.cpu().numpy())
        preds = np.vstack(preds)
        truths = np.vstack(truths)

        mse = mean_squared_error(truths, preds)
        r2 = r2_score(truths, preds)

        return mse, r2
    
    # Hyperparameter Optimization with Optuna
    # =======================
    def objective(trial):
        # hyperparameters to search
        hidden_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_dims = []
        for i in range(hidden_layers):
            if X_train.shape[1] <= 64:  # very low-dim
                units = trial.suggest_int(f"n_units_l{i}", 16, 128, step=16)
            elif X_train.shape[1] <= 512:  # medium
                units = trial.suggest_int(f"n_units_l{i}", 64, 512, step=32)
            else:  # large (768, 2048, ...)
                units = trial.suggest_int(f"n_units_l{i}", 128, 1024, step=64)
            hidden_dims.append(units)
        # hidden_dims = [trial.suggest_int(f"n_units_l{i}", 16, 512) for i in range(hidden_layers)]
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        # dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MLP(input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout=dropout)

        mse, r2 = train_model(model, train_loader, val_loader, epochs=30, lr=lr, device=device)
        trial.report(mse, step=0)

        return mse  # minimize MSE
    
    # Run Optuna search
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("Best hyperparameters:", study.best_params)
    print("Best MSE:", study.best_value)

    best_params = study.best_params
    hidden_layers = best_params["n_layers"]
    hidden_dims = [best_params[f"n_units_l{i}"] for i in range(hidden_layers)]
    dropout = best_params["dropout"]
    lr = best_params["lr"]
    batch_size = best_params["batch_size"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]
    final_model = MLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    mse, r2 = train_model(final_model, train_loader, val_loader, epochs=300, lr=lr, device=device)
    rmse = np.sqrt(mse)
    print(f"Final Test MSE: {mse:.4f}, R2: {r2:.4f}")

    # Collect predictions for saving 
    final_model.eval()
    with torch.no_grad():
        y_pred = y_scaler.inverse_transform(final_model(X_test_tensor.to(device)).cpu().numpy())
        y_true = y_scaler.inverse_transform(y_test)

    # Save results if save_dir provided
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        # Save predictions
        pred_df = pd.DataFrame({"y_true": y_true.flatten(), "y_pred": y_pred.flatten()})
        pred_csv = os.path.join(save_dir, "predictions.csv")
        pred_df.to_csv(pred_csv, index=False)
        print(f"Predictions saved to {pred_csv}")

        # Save best hyperparameters
        hp_file = os.path.join(save_dir, "best_hyperparameters.json")
        with open(hp_file, "w") as f:
            json.dump(best_params, f, indent=4)
        print(f"Best hyperparameters saved to {hp_file}")

        # Save scatter plot
        if plot:
            plt.figure(figsize=(6,6))
            plt.scatter(y_true, y_pred, alpha=0.6)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")  
            plt.xlabel("True pCMC")
            plt.ylabel("Predicted pCMC")
            plt.title("True vs Predicted pCMC")
            plot_file = os.path.join(save_dir, "scatter_plot.png")
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Plot saved to {plot_file}")
    else:
        # just show plot if not saving
        if plot:
            plt.figure(figsize=(6,6))
            plt.scatter(y_true, y_pred, alpha=0.6)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")  
            plt.xlabel("True pCMC")
            plt.ylabel("Predicted pCMC")
            plt.title("True vs Predicted pCMC")
            plt.show()

    return rmse, r2, y_pred, y_true

def run_NN_update(df, plot=False, seed=None, save_dir=None):

    # Load and Prepare Data
    # =======================
    end_column_index = df.columns.get_loc('Surfactant_Type')
    X = df.iloc[:, :end_column_index].values.astype(np.float32)  # features
    y = df["pCMC"].values.astype(np.float32).reshape(-1, 1)  # target

    # scale features
    x_scaler = StandardScaler()
    X = x_scaler.fit_transform(X)

    # scale target as well
    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # convert to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Define Model Class
    # =======================
    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dims, dropout=0.2):
            super(MLP, self).__init__()
            layers = []
            prev_dim = input_dim
            for h in hidden_dims:
                layers.append(nn.Linear(prev_dim, h))
                layers.append(nn.BatchNorm1d(h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev_dim = h
            layers.append(nn.Linear(prev_dim, 1))  # regression output
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)
        
    # Training Function
    # =======================
    def train_model(model, train_loader, val_loader, epochs, lr, device):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                        factor=0.5, patience=5, verbose=True)

        model.to(device)
        best_loss = float("inf")
        patience = 20
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            model.eval()
            val_preds, val_truths = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    val_preds.append(pred.cpu().numpy())
                    val_truths.append(yb.cpu().numpy())

            val_preds = np.vstack(val_preds)
            val_truths = np.vstack(val_truths)

            # RMSE on validation
            val_rmse = np.sqrt(mean_squared_error(val_truths, val_preds))

            # Scheduler step
            scheduler.step(val_rmse)

            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val RMSE: {val_rmse:.4f}")

            # Early stopping 
            if val_rmse < best_loss:
                best_loss = val_rmse
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Final evaluation on validation set
        model.eval()
        with torch.no_grad():
            preds, truths = [], []
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds.append(model(xb).cpu().numpy())
                truths.append(yb.cpu().numpy())
        preds = np.vstack(preds)
        truths = np.vstack(truths)

        mse = mean_squared_error(truths, preds)
        r2 = r2_score(truths, preds)

        return mse, r2
    
    # Hyperparameter Optimization with Optuna
    # =======================
    def objective(trial):
        # hyperparameters to search
        hidden_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_dims = []
        for i in range(hidden_layers):
            if X_train.shape[1] <= 64:  # very low-dim
                max_units = 128
                min_units = 16
                step = 8
            elif X_train.shape[1] <= 512:  # medium
                max_units = 512
                min_units = 64
                step = 32
            elif X_train.shape[1] <= 1024:  # large (500–1000 features)
                max_units = 1024
                min_units = 128
                step = 32
            else:  # large (768, 2048, ...)
                max_units = 1024
                min_units = 128
                step = 64

            # Decrease units with each deeper layer
            if i == 0:
                units = trial.suggest_int(f"n_units_l{i}", max_units // 2, max_units, step=step)
            else:
                prev_units = hidden_dims[-1]
                units = trial.suggest_int(
                    f"n_units_l{i}",
                    max(min_units, prev_units // 4),
                    max(min_units + step, prev_units // 2),
                    step=step
                )

            hidden_dims.append(units)

        dropout = trial.suggest_float("dropout", 0.2, 0.6)
        lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        # dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MLP(input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout=dropout)

        mse, r2 = train_model(model, train_loader, val_loader, epochs=30, lr=lr, device=device)
        trial.report(mse, step=0)

        return mse  # minimize MSE
    
    # Run Optuna search
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("Best hyperparameters:", study.best_params)
    print("Best MSE:", study.best_value)

    best_params = study.best_params
    hidden_layers = best_params["n_layers"]
    hidden_dims = [best_params[f"n_units_l{i}"] for i in range(hidden_layers)]
    dropout = best_params["dropout"]
    lr = best_params["lr"]
    batch_size = best_params["batch_size"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]
    final_model = MLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    mse, r2 = train_model(final_model, train_loader, val_loader, epochs=300, lr=lr, device=device)
    rmse = np.sqrt(mse)
    print(f"Final Test MSE: {mse:.4f}, R2: {r2:.4f}")

    # Collect predictions for saving
    final_model.eval()
    with torch.no_grad():
        y_pred = y_scaler.inverse_transform(final_model(X_test_tensor.to(device)).cpu().numpy())
        y_true = y_scaler.inverse_transform(y_test)

    # Save results if save_dir provided
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        # Save predictions
        pred_df = pd.DataFrame({"y_true": y_true.flatten(), "y_pred": y_pred.flatten()})
        pred_csv = os.path.join(save_dir, "predictions.csv")
        pred_df.to_csv(pred_csv, index=False)
        print(f"Predictions saved to {pred_csv}")

        # Save best hyperparameters
        hp_file = os.path.join(save_dir, "best_hyperparameters.json")
        with open(hp_file, "w") as f:
            json.dump(best_params, f, indent=4)
        print(f"Best hyperparameters saved to {hp_file}")

        # Save scatter plot
        if plot:
            plt.figure(figsize=(6,6))
            plt.scatter(y_true, y_pred, alpha=0.6)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")  
            plt.xlabel("True pCMC")
            plt.ylabel("Predicted pCMC")
            plt.title("True vs Predicted pCMC")
            plot_file = os.path.join(save_dir, "scatter_plot.png")
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Plot saved to {plot_file}")
    else:
        # just show plot if not saving
        if plot:
            plt.figure(figsize=(6,6))
            plt.scatter(y_true, y_pred, alpha=0.6)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")  
            plt.xlabel("True pCMC")
            plt.ylabel("Predicted pCMC")
            plt.title("True vs Predicted pCMC")
            plt.show()

    return rmse, r2, y_pred, y_true