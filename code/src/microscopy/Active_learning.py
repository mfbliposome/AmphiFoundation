import numpy as np
import torch
import gpytorch
from scipy.stats import entropy
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import time

from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement, qProbabilityOfImprovement, qUpperConfidenceBound
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from botorch.sampling import SobolQMCNormalSampler
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.models import ExactGP
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior
from gpytorch.constraints import GreaterThan


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

import pickle
import scipy.stats as stats

# Regression model

def objective_function(data, threshold=1):
    # Define the concentrations of amphiphiles
    c1, c2, c3, c4, c5, c6, c7 = data.iloc[:, 0:7].values.T
    # Calculate the sum of amphiphile concentrations
    sum_concentrations = c1 + c2 + c3 + c4 + c5 + c6 + c7
    vesicle_num = data.iloc[:, -1].values
    
    # If vesicle area is below threshold, penalize heavily
    below_threshold = vesicle_num < threshold
    sum_concentrations[below_threshold] += np.log(100)  # Adding log(100) penalizes heavily
    
    # Return the transformed sum of amphiphile concentrations
    return -sum_concentrations # in order to minimize function, in botorch, maximize is defalut

def build_model(init_x, init_y, priors):
    '''
    Build the Gaussian process model with customize priors
    '''

    covar_module = ScaleKernel(
        base_kernel=MaternKernel(
            nu=priors['kernel_smooth'],
            ard_num_dims=init_x.shape[1],
            # batch_shape=torch.Size(batch_shape),
            lengthscale_prior=GammaPrior(*priors['lengthscale']),
        ),
        # batch_shape=torch.Size(batch_shape),
        outputscale_prior=GammaPrior(*priors['outputscale']),
    )
    
    model_bo = SingleTaskGP(train_X=init_x, train_Y=init_y,        
                       input_transform=Normalize(d=7),
                       outcome_transform=Standardize(m=1),
                       covar_module=covar_module,
                           )
    
    mll = ExactMarginalLogLikelihood(model_bo.likelihood, model_bo)
    fit_gpytorch_mll(mll)
    
    return model_bo

def get_candidates(init_x, init_y, best_init_y, model, bounds, batch_size, acq_type='EI'):
    '''
    Using Bayesian Optimization to get next candidates
    '''
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]), seed=42)
    
    if acq_type == 'EI':
        acq_func = qExpectedImprovement(model, best_f=best_init_y, sampler=sampler)
    elif acq_type == 'UCB':
        acq_func = qUpperConfidenceBound(model, beta=1., sampler=sampler)
    elif acq_type == 'PI':
        acq_func = qProbabilityOfImprovement(model, best_f=best_init_y, sampler=sampler)
    else:
        print("Error: Please specify one of the following acquisition functions: 'EI', 'UCB', or 'PI'")
    # EI = ExpectedImprovement(model_bo, best_f=best_value, maximize=False)
    # ExpectedImprovement is an analytic acquisition function and only operates on single (q=1) points
    
    new_point_mc, ac_values = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=batch_size,
            num_restarts=20,
            raw_samples=100,
            options={},
        )
    
    # see training performance
    y_true = init_y.flatten()
    y_pred = model.posterior(init_x).mean.flatten()
    
    r2 = r2_score(y_true.detach().numpy(), y_pred.detach().numpy())
    mse = mean_squared_error(y_true.detach().numpy(), y_pred.detach().numpy())
    mae = mean_absolute_error(y_true.detach().numpy(), y_pred.detach().numpy())
    
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'R-squared (R2) score: {r2}')
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true.detach().numpy(), y_pred.detach().numpy(), color='blue', label='True vs. Predicted')
    plt.plot([min(y_true.detach().numpy()), max(y_true.detach().numpy())], [min(y_true.detach().numpy()), max(y_true.detach().numpy())], color='red', linestyle='--', label='Ideal')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs. Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.hist(y_true.detach().numpy(), bins=30, color='blue', alpha=0.7, label='True Values')
    plt.hist(y_pred.detach().numpy(), bins=30, color='orange', alpha=0.7, label='Predicted Values')
    
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Distribution of True and Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.show()

    return new_point_mc, ac_values, model, acq_func

def get_next_points_EI(init_x, init_y, best_init_y, priors, bounds, batch_size=48):
    '''
    Using Bayesian Optimization to get next candidates by using EI policy
    '''
    covar_module = ScaleKernel(
        base_kernel=MaternKernel(
            nu=priors['kernel_smooth'],
            ard_num_dims=init_x.shape[1],
            # batch_shape=torch.Size(batch_shape),
            lengthscale_prior=GammaPrior(*priors['lengthscale']),
        ),
        # batch_shape=torch.Size(batch_shape),
        outputscale_prior=GammaPrior(*priors['outputscale']),
    )
    
    model_bo = SingleTaskGP(train_X=init_x, train_Y=init_y,        
                       input_transform=Normalize(d=7),
                       outcome_transform=Standardize(m=1),
                       covar_module=covar_module,
                           )
    
    mll = ExactMarginalLogLikelihood(model_bo.likelihood, model_bo)
    fit_gpytorch_mll(mll)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]), seed=42)
        
    MC_EI = qExpectedImprovement(model_bo, best_f=best_init_y, sampler=sampler)
    # EI = ExpectedImprovement(model_bo, best_f=best_value, maximize=False)
    # ExpectedImprovement is an analytic acquisition function and only operates on single (q=1) points
    
    new_point_mc, ac_values = optimize_acqf(
            acq_function=MC_EI,
            bounds=bounds,
            q=batch_size,
            num_restarts=20,
            raw_samples=100,
            options={},
        )
    
    # see training performance
    y_true = init_y.flatten()
    y_pred = model_bo.posterior(init_x).mean.flatten()
    
    r2 = r2_score(y_true.detach().numpy(), y_pred.detach().numpy())
    mse = mean_squared_error(y_true.detach().numpy(), y_pred.detach().numpy())
    mae = mean_absolute_error(y_true.detach().numpy(), y_pred.detach().numpy())
    
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'R-squared (R2) score: {r2}')
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true.detach().numpy(), y_pred.detach().numpy(), color='blue', label='True vs. Predicted')
    plt.plot([min(y_true.detach().numpy()), max(y_true.detach().numpy())], [min(y_true.detach().numpy()), max(y_true.detach().numpy())], color='red', linestyle='--', label='Ideal')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs. Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.hist(y_true.detach().numpy(), bins=30, color='blue', alpha=0.7, label='True Values')
    plt.hist(y_pred.detach().numpy(), bins=30, color='orange', alpha=0.7, label='Predicted Values')
    
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Distribution of True and Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.show()

    return new_point_mc, ac_values, model_bo, MC_EI

def test_performance(df_input, objective_value, priors, bounds):
    '''
    See test results for model by random split dataset 
    '''
   
    start_time = time.time()


    # Number of samples
    Num=3
    # Define a list of threshold values to iterate over
    # threshold_values = [0.01, 0.02, 0.03, 0.04, 0.05]  
    threshold_values = [0.01]  

    # Initialize a dictionary to store overall performance for each threshold
    overall_performance_dict = {}

    # Iterate over each threshold value
    for threshold in threshold_values:
        R2 = []
        MSE = []
        MAE = []
        y_true = []
        y_pred = []
        
        for i in range(Num):
            # Split the dataset, leaving out the current sample
            X = df_input.iloc[:,0:7].values
            y = objective_value
        
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)
            init_x, init_y, best_init_y=X_train, y_train, y_train.max()
            init_x, init_y, best_init_y = torch.tensor(init_x), torch.tensor(init_y).unsqueeze(-1), torch.tensor(best_init_y).item()
            new_point_mc, ac_values, model, MC_EI = get_next_points_EI(init_x, init_y, best_init_y, priors, bounds, batch_size=48)
            
            pred_y = model.posterior(X=torch.tensor(X_test).unsqueeze(1)).mean.squeeze(-1)
            pred_y = pred_y.detach().numpy()
            true_y = y_test

            y_true.append(true_y)
            y_pred.append(pred_y)
            # flat arrays
            y_true_flat = np.concatenate(y_true)
            y_pred_flat = np.concatenate(y_pred).ravel()
        r2 = r2_score(y_true_flat, y_pred_flat)
        mse = mean_squared_error(y_true_flat, y_pred_flat)
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        print(f'Mean Squared Error (MSE): {mse}')
        print(f'Mean Absolute Error (MAE): {mae}')
        print(f'R-squared (R2) score: {r2}')

    end_time = time.time()
    running_time = end_time - start_time

    print("Script execution time:", running_time/60, "min")

def cal_entropy(data):
    '''
    Calculating the entropies for probability prediction
    '''

    entropies = []
    for row in data:
        # Replace 0 values with a small non-zero value (e.g., machine epsilon)
        row_nonzero = np.where(row == 0, np.finfo(float).eps, row)
        
        # Calculate entropy for the current row
        entropy_value = entropy(row_nonzero, base=2)
        
        # Append the entropy value to the list
        entropies.append(entropy_value)

    return np.array(entropies)


# Classification model
def cal_prob(df, model, batch_size=100000):
    '''
    Calculates the predicted probabilities for a given data set using a provided model in batches.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the features for prediction.
    model (scikit-learn estimator): A trained Gaussian Process Classifier.
    batch_size (int, optional): The number of samples to process in each batch. Default is 100,000.

    Returns:
    numpy.ndarray: An array of predicted probabilities for each sample in the input data.

    Application:
    This function is useful for making predictions on large datasets where processing all samples at once would be memory-intensive. 
    By breaking the data into smaller batches, 
    it ensures that predictions can be made efficiently and within memory constraints. 
    The function measures and prints the total running time for the prediction process.

    Example:
    Suppose you have a large dataset `df` and a trained model `model`, 
    you can use this function to get the predicted probabilities as follows:
    
    ```python
    probabilities = cal_prob(df, model, batch_size=50000)
    ```
    '''
    num_batches = len(df) // batch_size

    pred_probs = []

    start_time = time.time()

    for i in range(num_batches + 1):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        
        if start_idx >= len(df):
            break
        
        batch = df.values[start_idx:end_idx]
        pred_probs_batch = model.predict_proba(batch)
        pred_probs.extend(pred_probs_batch)

    # Convert the list of probabilities back to a numpy array if needed
    pred_probs = np.array(pred_probs)

    end_time = time.time()

    running_time = end_time - start_time
    print("Running time:", running_time, "seconds")

    return pred_probs

def cal_vesicles_per(pred_probs):
    '''
    Calculate the percent of samples of finding vesicles
    '''
    # Extract the second column
    second_column = pred_probs[:, 1]
    plt.hist(second_column)
    
    # Transform values: >0.5 to 1, else 0
    transformed = np.where(second_column >= 0.5, 1, 0)
    
    # Calculate the percentage of 1s
    percentage_of_ones = np.mean(transformed) * 100
    
    print(f"Percentage of 1s: {percentage_of_ones:.4f}%")
    return percentage_of_ones
    
# def plot_samples(df, title, color_map):
#     '''
#     Function to plot horizontal bars for each sample composition
#     '''
#     num_samples = df.shape[0]
#     features = df.columns
# 
#     plt.figure(figsize=(15, num_samples * 0.5))
# 
#     for i in range(num_samples):
#         sample = df.iloc[i, :]
#         left = 0
#         for feature in features:
#             plt.barh(i, sample[feature], left=left, color=color_map[feature], label=feature if i == 0 else "")
#             left += sample[feature]
# 
#     plt.xlabel('Concentration (mM)')
#     plt.ylabel('Sample Index')
#     plt.title(title)
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
#     plt.tight_layout()
#     plt.show()

def histogram_log(probs):
    '''
    Plot histogram with log scale (base=10)
    '''

    second_column = probs[:, 1]
    # base = 10
    plt.hist(second_column, log=True)


def generate_samples(bounds, limit, num_required_samples=10000000, num_generate_samples = 1000000):

    """
    Generates a specified number of valid samples within given bounds while 
    adhering to a concentration limit constraint. The function uses adaptive 
    sampling to improve efficiency and reduce computation time.

    Parameters:
    - bounds (ndarray): A 2xN array where the first row contains the lower 
      bounds and the second row contains the upper bounds for each dimension.
    - limit (float): The maximum allowable sum of the concentrationss for a sample to 
      be considered valid.
    - num_required_samples (int, optional): The total number of valid samples 
      required. Defaults to 10,000,000.
    - num_generate_samples (int, optional): The number of samples to generate 
      in each batch. Defaults to 1,000,000.

    Returns:
    - valid_samples1 (ndarray): An array of valid samples transformed with 
      log1p.
    """

    # Initial bounds
    original_bounds = bounds
    num_required_samples = 10000000
    num_dimensions = original_bounds.shape[1]

    # Define a simpler distribution (e.g., uniform) covering the initial bounds
    low = original_bounds[0]
    high = original_bounds[1]

    # Parameters for adaptive bounds
    adjust_factor = 0.9  # Factor to adjust the bounds by
    min_bound = 1e-4  # Minimum bound size to avoid too small sampling ranges

    start_time = time.time()  # Record start time

    valid_samples = []
    num_generate_samples = 1000000  # Initial batch size

    while len(valid_samples) < num_required_samples:
        # Generate samples
        samples = np.random.uniform(low=low, high=high, size=(num_generate_samples, num_dimensions))

        # Calculate the sum of the features
        total_concentration = samples.sum(axis=1)

        # Reject samples violating the constraint (total_concentration < limit)
        valid_samples_batch = samples[total_concentration < limit]

        # Append valid samples
        valid_samples.extend(valid_samples_batch.tolist())

        # Adjust the sampling bounds if the success rate is low
        if len(valid_samples_batch) < num_generate_samples * 0.01:
            range_size = high - low
            new_range_size = range_size * adjust_factor
            # Ensure the bounds do not get too small
            new_range_size = np.maximum(new_range_size, min_bound)
            high = low + new_range_size
            print(f"Adjusting bounds to {high}")

        # Print progress
        print(f"Generated {num_generate_samples} samples, found {len(valid_samples_batch)} valid samples")

    # Ensure we have exactly the required number of samples
    valid_samples = np.array(valid_samples[:num_required_samples])

    # Apply log1p transform
    valid_samples1 = np.log1p(valid_samples)
    print(valid_samples1.shape)

    end_time = time.time()  # Record end time

    # Calculate and print execution time
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")

    return valid_samples1

def plot_histogram(data, path, filename):
    '''
    data: The predicted probabilities array
    filename: The filename for saved figures, e.g. 'histogram_model.pdf'
    '''
    second_column = data[:, 1]
    # Create the histogram
    plt.figure(figsize=(12, 8))
    sns.histplot(second_column, bins=100, kde=False, color='royalblue')  # Increase the number of bins
    
    # Customize the plot
    plt.xlabel('Predicted Probabilities', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Save the plot with high resolution
    plt.savefig(path+filename, format='pdf', dpi=600, bbox_inches='tight')
    
    # Show the plot
    plt.show()