import json
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, mean_squared_error, r2_score
from sklearn.metrics import roc_auc_score, roc_curve, auc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import optuna

def run_NN_classification(df, plot=False, seed=None, save_dir=None):

    # Load and Prepare Data
    # =======================
    X = df.iloc[:, 0:-1].values.astype(np.float32)  # features
    y = df.iloc[:, -1].values.astype(np.int64)  # target (0/1 classification)

    # scale features only
    x_scaler = StandardScaler()
    X = x_scaler.fit_transform(X)

    # train/test split
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # within training data, split again for validation
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1875, random_state=seed)
    # (0.1875 × 0.8 ≈ 0.15) → total = 65/15/20 split

    # convert to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long) 
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)  
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Model
    # =======================
    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dims, dropout=0.2, num_classes=2):
            super(MLP, self).__init__()
            layers = []
            prev_dim = input_dim
            for h in hidden_dims:
                layers.append(nn.Linear(prev_dim, h))
                layers.append(nn.BatchNorm1d(h))   # normalize activations
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev_dim = h
            layers.append(nn.Linear(prev_dim, num_classes))  
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)  
    
    # Training
    # =======================
    def train_model(model, train_loader, val_loader, epochs, lr, device):
        criterion = nn.CrossEntropyLoss() 
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) # L2 regularization
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                        factor=0.5, patience=5, verbose=True)

        model.to(device)
        best_acc = 0.0
        patience = 20
        patience_counter = 0
        best_model_state = None
        best_val_loss = float('inf')  


        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            # Validation phase 
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    val_loss += loss.item() * xb.size(0)
            val_loss /= len(val_loader.dataset)

            scheduler.step(-val_loss)
            print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}")

            # Early stopping on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return best_acc

    # Optuna objective
    # =======================
    def objective(trial):
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

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MLP(input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout=dropout, num_classes=len(np.unique(y)))

        best_acc = train_model(model, train_loader, val_loader, epochs=30, lr=lr, device=device)
        return best_acc  
    
    # Run Optuna search
    study = optuna.create_study(direction="maximize") 
    study.optimize(objective, n_trials=20)

    print("Best hyperparameters:", study.best_params)
    print("Best Val Accuracy:", study.best_value)

    # Train final model with best params
    best_params = study.best_params
    hidden_layers = best_params["n_layers"]
    hidden_dims = [best_params[f"n_units_l{i}"] for i in range(hidden_layers)]
    dropout = best_params["dropout"]
    lr = best_params["lr"]
    batch_size = best_params["batch_size"]
    # Combine train + val for final training
    X_train_full = np.concatenate([X_train, X_val], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)
    train_full_dataset = TensorDataset(torch.tensor(X_train_full, dtype=torch.float32),
                                    torch.tensor(y_train_full, dtype=torch.long))
    train_loader = DataLoader(train_full_dataset, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_model = MLP(input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout=dropout, num_classes=len(np.unique(y)))
    best_acc = train_model(final_model, train_loader, test_loader, epochs=300, lr=lr, device=device)

    # Evaluate on test
    final_model.eval()
    with torch.no_grad():
        logits = final_model(X_test_tensor.to(device))
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_true = y_test

    test_acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    # print(f"Final Test Accuracy: {test_acc:.4f}, F1: {f1:.4f}")
    y_prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    roc_auc = roc_auc_score(y_true, y_prob)
    print(f"Final Test Accuracy: {test_acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

    # Save results if save_dir provided 
    if save_dir is not None:
        
        sub_dir = f"run_{seed}"
        save_dir = os.path.join(save_dir, sub_dir)
        os.makedirs(save_dir, exist_ok=True)

        # Save predictions
        pred_df = pd.DataFrame({"y_true": y_true.flatten(), "y_pred": y_pred.flatten(), "y_prob": y_prob.flatten()})
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
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            plt.figure()
            plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=300, bbox_inches="tight")
            plt.close()

            cm = confusion_matrix(y_true, y_pred)
            ConfusionMatrixDisplay(cm).plot(cmap="Blues")
            plt.title("Confusion Matrix")
            plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
            plt.show()
    else:
        # just show plot if not saving
        if plot:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            plt.figure()
            plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=300, bbox_inches="tight")
            plt.close()
            # Plot confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            ConfusionMatrixDisplay(cm).plot(cmap="Blues")
            plt.title("Confusion Matrix")
            plt.show()
  
    return roc_auc, test_acc, f1, y_pred, y_true, y_prob, final_model

def run_NN_classification_update(df, plot=False, seed=None, save_dir=None):

    # Load and Prepare Data
    # =======================
    X = df.iloc[:, 0:-1].values.astype(np.float32)  # features
    y = df.iloc[:, -1].values.astype(np.int64)  # target (0/1 classification)

    # scale features only
    x_scaler = StandardScaler()
    X = x_scaler.fit_transform(X)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # convert to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)   
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Model
    # =======================
    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dims, dropout=0.2, num_classes=2):
            super(MLP, self).__init__()
            layers = []
            prev_dim = input_dim
            for h in hidden_dims:
                layers.append(nn.Linear(prev_dim, h))
                layers.append(nn.BatchNorm1d(h))   # normalize activations
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev_dim = h
            layers.append(nn.Linear(prev_dim, num_classes))  
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)  
    
    # Training
    # =======================
    def train_model(model, train_loader, val_loader, epochs, lr, device):
        criterion = nn.CrossEntropyLoss() 
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) # L2 regularization
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                        factor=0.5, patience=5, verbose=True)

        model.to(device)
        best_acc = 0.0
        patience = 20
        patience_counter = 0
        best_model_state = None
        best_val_loss = float('inf')  


        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            # Validation phase 
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    val_loss += loss.item() * xb.size(0)
            val_loss /= len(val_loader.dataset)

            scheduler.step(-val_loss)
            print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}")

            # Early stopping on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return best_acc

    # Optuna objective
    # =======================
    def objective(trial):
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

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MLP(input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout=dropout, num_classes=len(np.unique(y)))

        best_acc = train_model(model, train_loader, val_loader, epochs=30, lr=lr, device=device)
        return best_acc  
    
    # Run Optuna search
    study = optuna.create_study(direction="maximize") 
    study.optimize(objective, n_trials=20)

    print("Best hyperparameters:", study.best_params)
    print("Best Val Accuracy:", study.best_value)

    # Train final model with best params
    best_params = study.best_params
    hidden_layers = best_params["n_layers"]
    hidden_dims = [best_params[f"n_units_l{i}"] for i in range(hidden_layers)]
    dropout = best_params["dropout"]
    lr = best_params["lr"]
    batch_size = best_params["batch_size"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_model = MLP(input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout=dropout, num_classes=len(np.unique(y)))
    best_acc = train_model(final_model, train_loader, val_loader, epochs=300, lr=lr, device=device)

    # Evaluate on test
    final_model.eval()
    with torch.no_grad():
        logits = final_model(X_test_tensor.to(device))
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_true = y_test

    test_acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    # print(f"Final Test Accuracy: {test_acc:.4f}, F1: {f1:.4f}")
    y_prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    roc_auc = roc_auc_score(y_true, y_prob)
    print(f"Final Test Accuracy: {test_acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

    # Save results if save_dir provided 
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        # Save predictions
        pred_df = pd.DataFrame({"y_true": y_true.flatten(), "y_pred": y_pred.flatten(), "y_prob": y_prob.flatten()})
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
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            plt.figure()
            plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=300, bbox_inches="tight")
            plt.close()

            cm = confusion_matrix(y_true, y_pred)
            ConfusionMatrixDisplay(cm).plot(cmap="Blues")
            plt.title("Confusion Matrix")
            plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
            plt.show()
    else:
        # just show plot if not saving
        if plot:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            plt.figure()
            plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=300, bbox_inches="tight")
            plt.close()
            # Plot confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            ConfusionMatrixDisplay(cm).plot(cmap="Blues")
            plt.title("Confusion Matrix")
            plt.show()
  

    return roc_auc, test_acc, f1, y_pred, y_true, y_prob, final_model
