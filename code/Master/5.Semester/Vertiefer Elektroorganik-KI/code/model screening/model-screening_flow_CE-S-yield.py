import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge, HuberRegressor
from sklearn.neural_network import MLPRegressor 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score, ParameterGrid
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import numpy as np
np.int = int
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, RegressorMixin
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#read input data
df_input = pd.read_csv('flow-cell.csv', index_col=0)

#filter input data
indices_to_remove_input = ['FE']
filtered_input = df_input.drop(indices_to_remove_input)
filtered_input = filtered_input.fillna(0) # fill  empty cells with 0
filtered_input = filtered_input.transpose() # change positions of columns and rows
filtered_input = filtered_input.loc[:, :'t [min]'] # remove columns after certain column
filtered_input = filtered_input.drop(['21065', '21090', '21091', '21097', '21092', '21080']) # remove outlier
filtered_input = filtered_input.drop(['21025', '21026', '21027', '21030', '21036']) # remove data without CE
#print(filtered_input)

#filter output data
df_output = pd.read_csv('flow-cell.csv', index_col=0)
filtered_output = df_output.loc['CE fuel [%]':, :,] # remove columns after certain column
filtered_output = filtered_output.fillna(0) # fill  empty cells with 0
filtered_output = filtered_output.transpose() # change positions of columns and rows
filtered_output = filtered_output.drop(['21065', '21090', '21091', '21097', '21092', '21080']) # remove outlier
filtered_output = filtered_output.drop(['21025', '21026', '21027', '21030', '21036']) # remove data without CE
#print(filtered_output)

X = filtered_input.values
Y = filtered_output[['CE fuel [%]', 'S decane w/o gas [%]','Ydecane total acid [%]']].values
#print(Y)

def standardize(X_train, X_test):
#def standardize(X_train, X_test, Y_train, Y_test):
    scaler = StandardScaler() # Initialize the StandardScaler
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    #Y_train = scaler.fit_transform(Y_train)
    #Y_test = scaler.transform(Y_test)
    return X_train, X_test
    #return X_train, X_test, Y_train, Y_test

def normalize(X_train, X_test):
    scaler = MinMaxScaler() # Initialize the MinMaxScaler
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    #Y_train = scaler.fit_transform(Y_train)
    #Y_test = scaler.transform(Y_test)
    return X_train, X_test

means_mse_train = []
means_rmse_train = []
means_r2_train = []
stds_mse_train = []
stds_rmse_train = []
stds_r2_train = []

means_mse = []
means_rmse = []
means_r2 = []
stds_mse = []
stds_rmse = []
stds_r2 = []

# gpr
random_state_counter_init = 42
random_state_counter = random_state_counter_init
runs = 10
#MSE_test_sum = 0
#RMSE_test_sum = 0
#R2_test_sum = 0
mse_train_values = []
rmse_train_values = []
r2_train_values = []

mse_values = []
rmse_values = []
r2_values = []

best_hypam1_values = []
best_hypam2_values = []

hyperparameters = ["alpha", "length scale"]

columns = []

while random_state_counter < (random_state_counter_init + runs):
    # split data into training and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state_counter)

    X_train, X_test=standardize(X_train, X_test)
    X_train, X_test=normalize(X_train, X_test)

    # Bayesian optimization GPR

    # Define the Gaussian regression model with hyperparameters to optimize
    class GaussianRegression(BaseEstimator):
        def __init__(self, alpha=1.0, kernel_length_scale=1.0):
            self.alpha = alpha
            self.kernel_length_scale = kernel_length_scale

        def fit(self, X, Y):
            kernel = RBF(length_scale=self.kernel_length_scale)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=self.alpha)
            self.model = make_pipeline(StandardScaler(), gp)
            self.model.fit(X, Y)
            return self

        def predict(self, X):
            return self.model.predict(X)

    # Define the objective function for optimization
    def objective(params):
        alpha = params['alpha']
        kernel_length_scale = params['kernel_length_scale']

        model = GaussianRegression(alpha=alpha, kernel_length_scale=kernel_length_scale)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        return -mean_squared_error(Y_test, Y_pred)  # Minimize MSE

    # Define the search space for hyperparameters
    param_space = {
        'alpha': Real(1e-1, 1e2, prior='log-uniform'),
        'kernel_length_scale': Real(1e-1, 1e2, prior='log-uniform')
    }

    # Initialize Bayesian optimization
    opt = BayesSearchCV(GaussianRegression(), param_space, n_iter=50, cv=5, n_jobs=-1, scoring='neg_mean_squared_error', random_state=random_state_counter)

    # Perform Bayesian optimization
    opt.fit(X_train, Y_train)

    # Get the best hyperparameters found by the optimizer
    best_params = opt.best_params_
    best_alpha = best_params['alpha']
    best_length_scale = best_params['kernel_length_scale']


    # for bayesian optimization
    best_gpr_bo = GaussianRegression(alpha=best_alpha, kernel_length_scale=best_length_scale)
    best_gpr_bo.fit(X_train, Y_train)


    Y_predict_train_bo = best_gpr_bo.predict(X_train)
    Y_predict_test_bo = best_gpr_bo.predict(X_test) 

    mse_training_bo = mean_squared_error(Y_train, Y_predict_train_bo)
    rmse_training_bo = mean_squared_error(Y_train, Y_predict_train_bo, squared=False)
    r2_training_bo = r2_score(Y_train, Y_predict_train_bo)

    mse_test_bo = mean_squared_error(Y_test, Y_predict_test_bo)
    rmse_test_bo = mean_squared_error(Y_test, Y_predict_test_bo, squared=False)
    r2_test_bo = r2_score(Y_test, Y_predict_test_bo)
    
    columns.append(random_state_counter)
    random_state_counter += 1
    #MSE_test_sum += mse_test_bo
    #RMSE_test_sum += rmse_test_bo
    #R2_test_sum += r2_test_bo
    mse_train_values.append(mse_training_bo)
    rmse_train_values.append(rmse_training_bo)
    r2_train_values.append(r2_training_bo)
    
    mse_values.append(mse_test_bo)
    rmse_values.append(rmse_test_bo)
    r2_values.append(r2_test_bo)
    best_hypam1_values.append(best_alpha)
    best_hypam2_values.append(best_length_scale)
    

#MSE_test_avg = MSE_test_sum / runs
#RMSE_test_avg = RMSE_test_sum / runs
#R2_test_avg = R2_test_sum / runs
mean_mse_train = np.mean(mse_train_values)
mean_rmse_train = np.mean(rmse_train_values)
mean_r2_train = np.mean(r2_train_values)
mean_mse = np.mean(mse_values)
mean_rmse = np.mean(rmse_values)
mean_r2 = np.mean(r2_values)
mean_best_hypam1 = np.mean(best_hypam1_values)
mean_best_hypam2 = np.mean(best_hypam2_values)

std_mse_train = np.std(mse_train_values)
std_rmse_train = np.std(rmse_train_values)
std_r2_train = np.std(r2_train_values)
std_mse = np.std(mse_values)
std_rmse = np.std(rmse_values)
std_r2 = np.std(r2_values)
std_best_hypam1 = np.std(best_hypam1_values)
std_best_hypam2 = np.std(best_hypam2_values)

mse_train_values.append(mean_mse_train)
mse_train_values.append(std_mse_train)
rmse_train_values.append(mean_rmse_train)
rmse_train_values.append(std_rmse_train)
r2_train_values.append(mean_r2_train)
r2_train_values.append(std_r2_train)
mse_values.append(mean_mse)
mse_values.append(std_mse)
rmse_values.append(mean_rmse)
rmse_values.append(std_rmse)
r2_values.append(mean_r2)
r2_values.append(std_r2)
best_hypam1_values.append(mean_best_hypam1)
best_hypam1_values.append(std_best_hypam1)
best_hypam2_values.append(mean_best_hypam2)
best_hypam2_values.append(std_best_hypam2)

means_mse_train.append(mean_mse_train)
stds_mse_train.append(std_mse_train)
means_rmse_train.append(mean_rmse_train)
stds_rmse_train.append(std_rmse_train)
means_r2_train.append(mean_r2_train)
stds_r2_train.append(std_r2_train)
means_mse.append(mean_mse)
stds_mse.append(std_mse)
means_rmse.append(mean_rmse)
stds_rmse.append(std_rmse)
means_r2.append(mean_r2)
stds_r2.append(std_r2)

print("average MSE on test set:", mean_mse)
print("Standard Deviation of MSE:", std_mse)
print("average RMSE on test set:", mean_rmse)
print("Standard Deviation of RMSE:", std_rmse)
print("average R2 on test set:", mean_r2)
print("Standard Deviation of R2:", std_r2)

summary = np.array([mse_train_values, rmse_train_values, r2_train_values, mse_values, rmse_values, r2_values, best_hypam1_values, best_hypam2_values])
columns.append("mean")
columns.append("std")
df_summary = pd.DataFrame(summary, columns=columns)
index_values = ['mse_train', 'rmse_train', 'r2_train', 'mse_test', 'rmse_test', 'r2_test', hyperparameters[0], hyperparameters[1]] 
df_summary = df_summary.set_index(pd.Index(index_values))
print(df_summary)
df_summary.to_csv('analytics_summary_gpr.csv', index=True)

# krr
random_state_counter_init = 42
random_state_counter = random_state_counter_init
runs = 10
#MSE_test_sum = 0
#RMSE_test_sum = 0
#R2_test_sum = 0
mse_train_values = []
rmse_train_values = []
r2_train_values = []

mse_values = []
rmse_values = []
r2_values = []

best_hypam1_values = []
best_hypam2_values = []
best_hypam3_values = []

hyperparameters = ["alpha", "gamma", "kernel"]

columns = []

while random_state_counter < (random_state_counter_init + runs):
    # split data into training and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state_counter)

    # standardize data
    scaler = StandardScaler() # Initialize the StandardScaler
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # normalize data with min-max scaling
    scaler = MinMaxScaler() # Initialize the MinMaxScaler
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Bayesian optimization krr

    def objective(alpha, kernel, gamma, degree, coef0):
        # Create a KRR model with the specified hyperparameters
        krr = KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)

        # Fit the model to the training data
        krr.fit(X_train, Y_train)

        # Make predictions on the test data
        Y_predict_krr_bo = krr.predict(X_train)

        # Calculate the mean squared error
        mse = mean_squared_error(Y_train, Y_predict_krr_bo)

        return -mse  # We want to minimize MSE, so we use negative MSE for maximization
    
    # Define the search space for hyperparameters
    param_space = {
        'alpha': Real(1e-1, 1e2, prior='log-uniform'),
        'kernel': Categorical(['linear', 'rbf', 'sigmoid']),
        'gamma': Real(1e-6, 1e3, prior='log-uniform'),
        #'degree': Integer(2, 4),
        'coef0': Real(-1, 1, prior='uniform')
    } # poly overfitted too much
    
    # Initialize Bayesian optimization
    opt = BayesSearchCV(
        KernelRidge(),
        param_space,
        n_iter=60,  # Number of iterations
        cv=5,  # Number of cross-validation folds
        n_jobs=-1,  # Use all available CPU cores
        scoring='neg_mean_squared_error',  # Negative MSE for maximization
        random_state=random_state_counter)
        
    # Perform Bayesian optimization
    opt.fit(X_train, Y_train)

    # Get the best hyperparameters found by the optimizer
    best_params = opt.best_params_
    best_alpha = best_params['alpha']
    best_kernel = best_params['kernel']
    best_gamma = best_params['gamma']

    # Train the final KRR model with the best hyperparameters
    final_krr = KernelRidge(alpha=best_alpha, kernel=best_kernel, gamma=best_gamma)
    final_krr.fit(X_train, Y_train)

    # Evaluate the final model
    Y_predict_test_bo = final_krr.predict(X_test)
    Y_predict_train_bo = final_krr.predict(X_train)

    mse_training_bo = mean_squared_error(Y_train, Y_predict_train_bo)
    rmse_training_bo = mean_squared_error(Y_train, Y_predict_train_bo, squared=False)
    r2_training_bo = r2_score(Y_train, Y_predict_train_bo)

    mse_test_bo = mean_squared_error(Y_test, Y_predict_test_bo)
    rmse_test_bo = mean_squared_error(Y_test, Y_predict_test_bo, squared=False)
    r2_test_bo = r2_score(Y_test, Y_predict_test_bo)
    
    columns.append(random_state_counter)
    random_state_counter += 1
    #MSE_test_sum += mse_test_bo
    #RMSE_test_sum += rmse_test_bo
    #R2_test_sum += r2_test_bo
    mse_train_values.append(mse_training_bo)
    rmse_train_values.append(rmse_training_bo)
    r2_train_values.append(r2_training_bo)
    
    mse_values.append(mse_test_bo)
    rmse_values.append(rmse_test_bo)
    r2_values.append(r2_test_bo)
    best_hypam1_values.append(best_alpha)
    best_hypam2_values.append(best_gamma)
    best_hypam3_values.append(best_kernel)

#MSE_test_avg = MSE_test_sum / runs
#RMSE_test_avg = RMSE_test_sum / runs
#R2_test_avg = R2_test_sum / runs
mean_mse_train = np.mean(mse_train_values)
mean_rmse_train = np.mean(rmse_train_values)
mean_r2_train = np.mean(r2_train_values)
mean_mse = np.mean(mse_values)
mean_rmse = np.mean(rmse_values)
mean_r2 = np.mean(r2_values)
mean_best_hypam1 = np.mean(best_hypam1_values)
mean_best_hypam2 = np.mean(best_hypam2_values)

std_mse_train = np.std(mse_train_values)
std_rmse_train = np.std(rmse_train_values)
std_r2_train = np.std(r2_train_values)
std_mse = np.std(mse_values)
std_rmse = np.std(rmse_values)
std_r2 = np.std(r2_values)
std_best_hypam1 = np.std(best_hypam1_values)
std_best_hypam2 = np.std(best_hypam2_values)

mse_train_values.append(mean_mse_train)
mse_train_values.append(std_mse_train)
rmse_train_values.append(mean_rmse_train)
rmse_train_values.append(std_rmse_train)
r2_train_values.append(mean_r2_train)
r2_train_values.append(std_r2_train)
mse_values.append(mean_mse)
mse_values.append(std_mse)
rmse_values.append(mean_rmse)
rmse_values.append(std_rmse)
r2_values.append(mean_r2)
r2_values.append(std_r2)
best_hypam1_values.append(mean_best_hypam1)
best_hypam1_values.append(std_best_hypam1)
best_hypam2_values.append(mean_best_hypam2)
best_hypam2_values.append(std_best_hypam2)
best_hypam3_values.append("")
best_hypam3_values.append("")

means_mse_train.append(mean_mse_train)
stds_mse_train.append(std_mse_train)
means_rmse_train.append(mean_rmse_train)
stds_rmse_train.append(std_rmse_train)
means_r2_train.append(mean_r2_train)
stds_r2_train.append(std_r2_train)
means_mse.append(mean_mse)
stds_mse.append(std_mse)
means_rmse.append(mean_rmse)
stds_rmse.append(std_rmse)
means_r2.append(mean_r2)
stds_r2.append(std_r2)    
    
print("average MSE on test set:", mean_mse)
print("Standard Deviation of MSE:", std_mse)
print("average RMSE on test set:", mean_rmse)
print("Standard Deviation of RMSE:", std_rmse)
print("average R2 on test set:", mean_r2)
print("Standard Deviation of R2:", std_r2)

summary = np.array([mse_train_values, rmse_train_values, r2_train_values, mse_values, rmse_values, r2_values, best_hypam1_values, best_hypam2_values, best_hypam3_values])
columns.append("mean")
columns.append("std")
df_summary = pd.DataFrame(summary, columns=columns)
index_values = ['mse_train', 'rmse_train', 'r2_train', 'mse_test', 'rmse_test', 'r2_test', hyperparameters[0], hyperparameters[1], hyperparameters[2]] 
df_summary = df_summary.set_index(pd.Index(index_values))
print(df_summary)
df_summary.to_csv('analytics_summary_krr.csv', index=True)

# krr only linear
random_state_counter_init = 42
random_state_counter = random_state_counter_init
runs = 10
#MSE_test_sum = 0
#RMSE_test_sum = 0
#R2_test_sum = 0
mse_train_values = []
rmse_train_values = []
r2_train_values = []

mse_values = []
rmse_values = []
r2_values = []

best_hypam1_values = []
best_hypam2_values = []
best_hypam3_values = []

hyperparameters = ["alpha", "gamma", "kernel"]

columns = []

while random_state_counter < (random_state_counter_init + runs):
    # split data into training and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state_counter)

    # standardize data
    scaler = StandardScaler() # Initialize the StandardScaler
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # normalize data with min-max scaling
    scaler = MinMaxScaler() # Initialize the MinMaxScaler
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Bayesian optimization krr

    def objective(alpha):
        # Create a KRR model with the specified hyperparameters
        krr = KernelRidge(alpha=alpha, kernel=linear)

        # Fit the model to the training data
        krr.fit(X_train, Y_train)

        # Make predictions on the test data
        Y_predict_krr_bo = krr.predict(X_train)

        # Calculate the mean squared error
        mse = mean_squared_error(Y_train, Y_predict_krr_bo)

        return -mse  # We want to minimize MSE, so we use negative MSE for maximization
    
    # Define the search space for hyperparameters
    param_space = {
        'alpha': Real(1e-1, 1e2, prior='log-uniform'),
    } # poly overfitted too much
    
    # Initialize Bayesian optimization
    opt = BayesSearchCV(
        KernelRidge(),
        param_space,
        n_iter=60,  # Number of iterations
        cv=5,  # Number of cross-validation folds
        n_jobs=-1,  # Use all available CPU cores
        scoring='neg_mean_squared_error',  # Negative MSE for maximization
        random_state=random_state_counter)
        
    # Perform Bayesian optimization
    opt.fit(X_train, Y_train)

    # Get the best hyperparameters found by the optimizer
    best_params = opt.best_params_
    best_alpha = best_params['alpha']

    # Train the final KRR model with the best hyperparameters
    final_krr = KernelRidge(alpha=best_alpha)
    final_krr.fit(X_train, Y_train)

    # Evaluate the final model
    Y_predict_test_bo = final_krr.predict(X_test)
    Y_predict_train_bo = final_krr.predict(X_train)

    mse_training_bo = mean_squared_error(Y_train, Y_predict_train_bo)
    rmse_training_bo = mean_squared_error(Y_train, Y_predict_train_bo, squared=False)
    r2_training_bo = r2_score(Y_train, Y_predict_train_bo)

    mse_test_bo = mean_squared_error(Y_test, Y_predict_test_bo)
    rmse_test_bo = mean_squared_error(Y_test, Y_predict_test_bo, squared=False)
    r2_test_bo = r2_score(Y_test, Y_predict_test_bo)
    
    columns.append(random_state_counter)
    random_state_counter += 1
    #MSE_test_sum += mse_test_bo
    #RMSE_test_sum += rmse_test_bo
    #R2_test_sum += r2_test_bo
    mse_train_values.append(mse_training_bo)
    rmse_train_values.append(rmse_training_bo)
    r2_train_values.append(r2_training_bo)
    
    mse_values.append(mse_test_bo)
    rmse_values.append(rmse_test_bo)
    r2_values.append(r2_test_bo)
    best_hypam1_values.append(best_alpha)

#MSE_test_avg = MSE_test_sum / runs
#RMSE_test_avg = RMSE_test_sum / runs
#R2_test_avg = R2_test_sum / runs
mean_mse_train = np.mean(mse_train_values)
mean_rmse_train = np.mean(rmse_train_values)
mean_r2_train = np.mean(r2_train_values)
mean_mse = np.mean(mse_values)
mean_rmse = np.mean(rmse_values)
mean_r2 = np.mean(r2_values)
mean_best_hypam1 = np.mean(best_hypam1_values)

std_mse_train = np.std(mse_train_values)
std_rmse_train = np.std(rmse_train_values)
std_r2_train = np.std(r2_train_values)
std_mse = np.std(mse_values)
std_rmse = np.std(rmse_values)
std_r2 = np.std(r2_values)
std_best_hypam1 = np.std(best_hypam1_values)

mse_train_values.append(mean_mse_train)
mse_train_values.append(std_mse_train)
rmse_train_values.append(mean_rmse_train)
rmse_train_values.append(std_rmse_train)
r2_train_values.append(mean_r2_train)
r2_train_values.append(std_r2_train)
mse_values.append(mean_mse)
mse_values.append(std_mse)
rmse_values.append(mean_rmse)
rmse_values.append(std_rmse)
r2_values.append(mean_r2)
r2_values.append(std_r2)
best_hypam1_values.append(mean_best_hypam1)
best_hypam1_values.append(std_best_hypam1)

means_mse_train.append(mean_mse_train)
stds_mse_train.append(std_mse_train)
means_rmse_train.append(mean_rmse_train)
stds_rmse_train.append(std_rmse_train)
means_r2_train.append(mean_r2_train)
stds_r2_train.append(std_r2_train) 
means_mse.append(mean_mse)
stds_mse.append(std_mse)
means_rmse.append(mean_rmse)
stds_rmse.append(std_rmse)
means_r2.append(mean_r2)
stds_r2.append(std_r2)    
    
print("average MSE on test set:", mean_mse)
print("Standard Deviation of MSE:", std_mse)
print("average RMSE on test set:", mean_rmse)
print("Standard Deviation of RMSE:", std_rmse)
print("average R2 on test set:", mean_r2)
print("Standard Deviation of R2:", std_r2)

summary = np.array([mse_train_values, rmse_train_values, r2_train_values, mse_values, rmse_values, r2_values, best_hypam1_values])
columns.append("mean")
columns.append("std")
df_summary = pd.DataFrame(summary, columns=columns)
index_values = ['mse_train', 'rmse_train', 'r2_train', 'mse_test', 'rmse_test', 'r2_test', hyperparameters[0]] 
df_summary = df_summary.set_index(pd.Index(index_values))
print(df_summary)
df_summary.to_csv('analytics_summary_krr_linear.csv', index=True)

# random forest
random_state_counter_init = 42
random_state_counter = random_state_counter_init
runs = 10
#MSE_test_sum = 0
#RMSE_test_sum = 0
#R2_test_sum = 0
mse_train_values = []
rmse_train_values = []
r2_train_values = []

mse_values = []
rmse_values = []
r2_values = []

best_hypam1_values = []
best_hypam2_values = []
best_hypam3_values = []
best_hypam4_values = []

hyperparameters = ["n_estimators", "max_depth", "min samples split", "min samples leaf"]

columns = []

while random_state_counter < (random_state_counter_init + runs):
    # split data into training and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state_counter)

    # standardize data
    scaler = StandardScaler() # Initialize the StandardScaler
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # normalize data with min-max scaling
    scaler = MinMaxScaler() # Initialize the MinMaxScaler
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Bayesian optimization krr

   # Define the objective function for optimization
    def objective_function(params):
        rf_regressor = RandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            random_state=42
        )

        rf_regressor.fit(X_train, Y_train)
        Y_pred = rf_regressor.predict(X_train)

        mse = mean_squared_error(Y_train, Y_pred)
        return mse
    
    param_space = {
        'n_estimators': Integer(1, 200),  # Number of trees
        'max_depth': Integer(1, 32),      # Maximum depth of trees
        'min_samples_split': Integer(2, 12),  # Minimum samples required to split an internal node
        'min_samples_leaf': Integer(1, 12),   # Minimum samples required in a leaf node
        }
    
    # Create a Bayesian Optimization object
    opt = BayesSearchCV(
        estimator=RandomForestRegressor(),  # Specify the estimator here
        search_spaces=param_space,
        scoring='neg_mean_squared_error',  # Negative MSE since we want to minimize it
        n_iter=60,  # Number of optimization iterations
        random_state=random_state_counter
    )

    # Fit the optimizer on your data
    opt.fit(X_train, Y_train)
    #opt.fit(X_train, Y_train.ravel())

    # Get the best hyperparameters found by the optimizer
    best_params = opt.best_params_
    best_n_estimators = best_params['n_estimators']
    best_max_depth = best_params['max_depth']
    best_min_samples_split = best_params['min_samples_split']
    best_min_samples_leaf = best_params['min_samples_leaf']
        
    final_rf = RandomForestRegressor(**best_params, random_state=42)
    final_rf.fit(X_train, Y_train)

    # Evaluate the final model
    Y_predict_test_bo = final_rf.predict(X_test)
    Y_predict_train_bo = final_rf.predict(X_train)

    mse_training_bo = mean_squared_error(Y_train, Y_predict_train_bo)
    rmse_training_bo = mean_squared_error(Y_train, Y_predict_train_bo, squared=False)
    r2_training_bo = r2_score(Y_train, Y_predict_train_bo)

    mse_test_bo = mean_squared_error(Y_test, Y_predict_test_bo)
    rmse_test_bo = mean_squared_error(Y_test, Y_predict_test_bo, squared=False)
    r2_test_bo = r2_score(Y_test, Y_predict_test_bo)
    
    columns.append(random_state_counter)
    random_state_counter += 1
    #MSE_test_sum += mse_test_bo
    #RMSE_test_sum += rmse_test_bo
    #R2_test_sum += r2_test_bo
    mse_train_values.append(mse_training_bo)
    rmse_train_values.append(rmse_training_bo)
    r2_train_values.append(r2_training_bo)
    
    mse_values.append(mse_test_bo)
    rmse_values.append(rmse_test_bo)
    r2_values.append(r2_test_bo)
    best_hypam1_values.append(best_n_estimators)
    best_hypam2_values.append(best_max_depth)
    best_hypam3_values.append(best_min_samples_split)
    best_hypam4_values.append(best_min_samples_leaf)

#MSE_test_avg = MSE_test_sum / runs
#RMSE_test_avg = RMSE_test_sum / runs
#R2_test_avg = R2_test_sum / runs
mean_mse_train = np.mean(mse_train_values)
mean_rmse_train = np.mean(rmse_train_values)
mean_r2_train = np.mean(r2_train_values)
mean_mse = np.mean(mse_values)
mean_rmse = np.mean(rmse_values)
mean_r2 = np.mean(r2_values)
mean_best_hypam1 = np.mean(best_hypam1_values)
mean_best_hypam2 = np.mean(best_hypam2_values)
mean_best_hypam3 = np.mean(best_hypam3_values)
mean_best_hypam4 = np.mean(best_hypam4_values)

std_mse_train = np.std(mse_train_values)
std_rmse_train = np.std(rmse_train_values)
std_r2_train = np.std(r2_train_values)
std_mse = np.std(mse_values)
std_rmse = np.std(rmse_values)
std_r2 = np.std(r2_values)
std_best_hypam1 = np.std(best_hypam1_values)
std_best_hypam2 = np.std(best_hypam2_values)
std_best_hypam3 = np.std(best_hypam3_values)
std_best_hypam4 = np.std(best_hypam4_values)

mse_train_values.append(mean_mse_train)
mse_train_values.append(std_mse_train)
rmse_train_values.append(mean_rmse_train)
rmse_train_values.append(std_rmse_train)
r2_train_values.append(mean_r2_train)
r2_train_values.append(std_r2_train)
mse_values.append(mean_mse)
mse_values.append(std_mse)
rmse_values.append(mean_rmse)
rmse_values.append(std_rmse)
r2_values.append(mean_r2)
r2_values.append(std_r2)
best_hypam1_values.append(mean_best_hypam1)
best_hypam1_values.append(std_best_hypam1)
best_hypam2_values.append(mean_best_hypam2)
best_hypam2_values.append(std_best_hypam2)
best_hypam3_values.append(mean_best_hypam3)
best_hypam3_values.append(std_best_hypam3)
best_hypam4_values.append(mean_best_hypam4)
best_hypam4_values.append(std_best_hypam4)

means_mse_train.append(mean_mse_train)
stds_mse_train.append(std_mse_train)
means_rmse_train.append(mean_rmse_train)
stds_rmse_train.append(std_rmse_train)
means_r2_train.append(mean_r2_train)
stds_r2_train.append(std_r2_train) 
means_mse.append(mean_mse)
stds_mse.append(std_mse)
means_rmse.append(mean_rmse)
stds_rmse.append(std_rmse)
means_r2.append(mean_r2)
stds_r2.append(std_r2)    
        
print("average MSE on test set:", mean_mse)
print("Standard Deviation of MSE:", std_mse)
print("average RMSE on test set:", mean_rmse)
print("Standard Deviation of RMSE:", std_rmse)
print("average R2 on test set:", mean_r2)
print("Standard Deviation of R2:", std_r2)

summary = np.array([mse_train_values, rmse_train_values, r2_train_values, mse_values, rmse_values, r2_values, best_hypam1_values, best_hypam2_values, best_hypam3_values, best_hypam4_values])
columns.append("mean")
columns.append("std")
df_summary = pd.DataFrame(summary, columns=columns)
index_values = ['mse_train', 'rmse_train', 'r2_train', 'mse_test', 'rmse_test', 'r2_test', hyperparameters[0], hyperparameters[1], hyperparameters[2], hyperparameters[3]] 
df_summary = df_summary.set_index(pd.Index(index_values))
print(df_summary)
df_summary.to_csv('analytics_summary_rf.csv', index=True)

# Decision tree
random_state_counter_init = 42
random_state_counter = random_state_counter_init
runs = 10

mse_train_values = []
rmse_train_values = []
r2_train_values = []

mse_values = []
rmse_values = []
r2_values = []

best_hypam1_values = []

hyperparameters = ["max_depth"]

columns = []

while random_state_counter < (random_state_counter_init + runs):
    # split data into training and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state_counter)

    # standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # normalize data with min-max scaling
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Bayesian optimization decision tree

    def objective_function(params):
        dt_regressor = DecisionTreeRegressor(
            max_depth=params['max_depth'],
            random_state=42
        )

        dt_regressor.fit(X_train, Y_train)
        Y_pred = dt_regressor.predict(X_train)

        mse = mean_squared_error(Y_train, Y_pred)
        return mse

    param_space = {
        'max_depth': Integer(1, 32),
    }

    opt = BayesSearchCV(
        estimator=DecisionTreeRegressor(),
        search_spaces=param_space,
        scoring='neg_mean_squared_error',
        n_iter=60,
        random_state=random_state_counter
    )

    opt.fit(X_train, Y_train)
    #opt.fit(X_train, Y_train.ravel())

    best_params = opt.best_params_
    best_max_depth = best_params['max_depth']

    final_dt = DecisionTreeRegressor(**best_params, random_state=42)
    final_dt.fit(X_train, Y_train)

    Y_predict_test_bo = final_dt.predict(X_test)
    Y_predict_train_bo = final_dt.predict(X_train)

    mse_training_bo = mean_squared_error(Y_train, Y_predict_train_bo)
    rmse_training_bo = mean_squared_error(Y_train, Y_predict_train_bo, squared=False)
    r2_training_bo = r2_score(Y_train, Y_predict_train_bo)

    mse_test_bo = mean_squared_error(Y_test, Y_predict_test_bo)
    rmse_test_bo = mean_squared_error(Y_test, Y_predict_test_bo, squared=False)
    r2_test_bo = r2_score(Y_test, Y_predict_test_bo)

    columns.append(random_state_counter)
    random_state_counter += 1

    mse_train_values.append(mse_training_bo)
    rmse_train_values.append(rmse_training_bo)
    r2_train_values.append(r2_training_bo)

    mse_values.append(mse_test_bo)
    rmse_values.append(rmse_test_bo)
    r2_values.append(r2_test_bo)
    best_hypam1_values.append(best_max_depth)

mean_mse_train = np.mean(mse_train_values)
mean_rmse_train = np.mean(rmse_train_values)
mean_r2_train = np.mean(r2_train_values)
mean_mse = np.mean(mse_values)
mean_rmse = np.mean(rmse_values)
mean_r2 = np.mean(r2_values)
mean_best_hypam1 = np.mean(best_hypam1_values)

std_mse_train = np.std(mse_train_values)
std_rmse_train = np.std(rmse_train_values)
std_r2_train = np.std(r2_train_values)
std_mse = np.std(mse_values)
std_rmse = np.std(rmse_values)
std_r2 = np.std(r2_values)
std_best_hypam1 = np.std(best_hypam1_values)

mse_train_values.append(mean_mse_train)
mse_train_values.append(std_mse_train)
rmse_train_values.append(mean_rmse_train)
rmse_train_values.append(std_rmse_train)
r2_train_values.append(mean_r2_train)
r2_train_values.append(std_r2_train)
mse_values.append(mean_mse)
mse_values.append(std_mse)
rmse_values.append(mean_rmse)
rmse_values.append(std_rmse)
r2_values.append(mean_r2)
r2_values.append(std_r2)
best_hypam1_values.append(mean_best_hypam1)
best_hypam1_values.append(std_best_hypam1)

means_mse_train.append(mean_mse_train)
stds_mse_train.append(std_mse_train)
means_rmse_train.append(mean_rmse_train)
stds_rmse_train.append(std_rmse_train)
means_r2_train.append(mean_r2_train)
stds_r2_train.append(std_r2_train) 
means_mse.append(mean_mse)
stds_mse.append(std_mse)
means_rmse.append(mean_rmse)
stds_rmse.append(std_rmse)
means_r2.append(mean_r2)
stds_r2.append(std_r2)    
    
print("average MSE on test set:", mean_mse)
print("Standard Deviation of MSE:", std_mse)
print("average RMSE on test set:", mean_rmse)
print("Standard Deviation of RMSE:", std_rmse)
print("average R2 on test set:", mean_r2)
print("Standard Deviation of R2:", std_r2)

summary = np.array([mse_train_values, rmse_train_values, r2_train_values, mse_values, rmse_values, r2_values, best_hypam1_values])
columns.append("mean")
columns.append("std")
df_summary = pd.DataFrame(summary, columns=columns)
index_values = ['mse_train', 'rmse_train', 'r2_train', 'mse_test', 'rmse_test', 'r2_test', hyperparameters[0]]
df_summary = df_summary.set_index(pd.Index(index_values))
print(df_summary)
df_summary.to_csv('analytics_summary_dt.csv', index=True)


# Multiple Linear Regression
random_state_counter_init = 42
random_state_counter = random_state_counter_init
runs = 10

mse_train_values = []
rmse_train_values = []
r2_train_values = []

mse_values = []
rmse_values = []
r2_values = []

best_hypam_values = []

hyperparameters = ["None"]

columns = []

while random_state_counter < (random_state_counter_init + runs):
    # split data into training and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state_counter)

    # standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # normalize data with min-max scaling
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    final_lr = LinearRegression()
    final_lr.fit(X_train, Y_train)

    Y_predict_test_bo = final_lr.predict(X_test)
    Y_predict_train_bo = final_lr.predict(X_train)

    mse_training_bo = mean_squared_error(Y_train, Y_predict_train_bo)
    rmse_training_bo = mean_squared_error(Y_train, Y_predict_train_bo, squared=False)
    r2_training_bo = r2_score(Y_train, Y_predict_train_bo)

    mse_test_bo = mean_squared_error(Y_test, Y_predict_test_bo)
    rmse_test_bo = mean_squared_error(Y_test, Y_predict_test_bo, squared=False)
    r2_test_bo = r2_score(Y_test, Y_predict_test_bo)

    columns.append(random_state_counter)
    random_state_counter += 1

    mse_train_values.append(mse_training_bo)
    rmse_train_values.append(rmse_training_bo)
    r2_train_values.append(r2_training_bo)

    mse_values.append(mse_test_bo)
    rmse_values.append(rmse_test_bo)
    r2_values.append(r2_test_bo)
    best_hypam_values.append(None)  # No hyperparameters for linear regression

mean_mse_train = np.mean(mse_train_values)
mean_rmse_train = np.mean(rmse_train_values)
mean_r2_train = np.mean(r2_train_values)
mean_mse = np.mean(mse_values)
mean_rmse = np.mean(rmse_values)
mean_r2 = np.mean(r2_values)

std_mse_train = np.std(mse_train_values)
std_rmse_train = np.std(rmse_train_values)
std_r2_train = np.std(r2_train_values)
std_mse = np.std(mse_values)
std_rmse = np.std(rmse_values)
std_r2 = np.std(r2_values)

mse_train_values.append(mean_mse_train)
mse_train_values.append(std_mse_train)
rmse_train_values.append(mean_rmse_train)
rmse_train_values.append(std_rmse_train)
r2_train_values.append(mean_r2_train)
r2_train_values.append(std_r2_train)
mse_values.append(mean_mse)
mse_values.append(std_mse)
rmse_values.append(mean_rmse)
rmse_values.append(std_rmse)
r2_values.append(mean_r2)
r2_values.append(std_r2)
best_hypam_values.append("None")

means_mse_train.append(mean_mse_train)
stds_mse_train.append(std_mse_train)
means_rmse_train.append(mean_rmse_train)
stds_rmse_train.append(std_rmse_train)
means_r2_train.append(mean_r2_train)
stds_r2_train.append(std_r2_train)
means_mse.append(mean_mse)
stds_mse.append(std_mse)
means_rmse.append(mean_rmse)
stds_rmse.append(std_rmse)
means_r2.append(mean_r2)
stds_r2.append(std_r2)    

print("average MSE on test set:", mean_mse)
print("Standard Deviation of MSE:", std_mse)
print("average RMSE on test set:", mean_rmse)
print("Standard Deviation of RMSE:", std_rmse)
print("average R2 on test set:", mean_r2)
print("Standard Deviation of R2:", std_r2)

summary = np.array([mse_train_values, rmse_train_values, r2_train_values, mse_values, rmse_values, r2_values])
columns.append("mean")
columns.append("std")
df_summary = pd.DataFrame(summary, columns=columns)
index_values = ['mse_train', 'rmse_train', 'r2_train', 'mse_test', 'rmse_test', 'r2_test']
df_summary = df_summary.set_index(pd.Index(index_values))
print(df_summary)
df_summary.to_csv('analytics_summary_lr.csv', index=True)


# xDNN
random_state_counter_init = 42
random_state_counter = random_state_counter_init
runs = 10

mse_train_values = []
rmse_train_values = []
r2_train_values = []

mse_values = []
rmse_values = []
r2_values = []

best_hypam1_values = []
best_hypam2_values = []
best_hypam3_values = []

hyperparameters = ["hidden_layer_sizes", "alpha", "learning_rate"]

columns = []

while random_state_counter < (random_state_counter_init + runs):
    # split data into training and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state_counter)

    # standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # normalize data with min-max scaling
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Bayesian optimization xDNN

    def objective_function(params):
        #hidden_layer_sizes = [int(x) for x in hidden_layer_sizes]):
        #hidden_layer_sizes = tuple(int(size) for size in params['hidden_layer_sizes'])
        mlp_regressor = MLPRegressor(
            hidden_layer_sizes=params['hidden_layer_sizes'],
            alpha=params['alpha'],
            learning_rate=params['learning_rate'],
            random_state=random_state_counter
        )

        mlp_regressor.fit(X_train, Y_train)
        Y_pred = mlp_regressor.predict(X_train)

        mse = mean_squared_error(Y_train, Y_pred)
        return mse

    param_space = {
        'hidden_layer_sizes': (50,250),
        'alpha': Real(1e-3, 1e1),
        'learning_rate': Categorical(["constant", "invscaling", "adaptive"])
    }

    opt = BayesSearchCV(
        estimator=MLPRegressor(),
        search_spaces=param_space,
        scoring='neg_mean_squared_error',
        n_iter=60,
        random_state=random_state_counter
    )

    opt.fit(X_train, Y_train)
    #opt.fit(X_train, Y_train.ravel())
    
    best_params = opt.best_params_
    best_hidden_layer_sizes = best_params['hidden_layer_sizes']
    best_alpha = best_params['alpha']
    best_learning_rate = best_params['learning_rate']

    final_mlp = MLPRegressor(**best_params, random_state=random_state_counter)
    final_mlp.fit(X_train, Y_train)

    Y_predict_test_bo = final_mlp.predict(X_test)
    Y_predict_train_bo = final_mlp.predict(X_train)

    mse_training_bo = mean_squared_error(Y_train, Y_predict_train_bo)
    rmse_training_bo = mean_squared_error(Y_train, Y_predict_train_bo, squared=False)
    r2_training_bo = r2_score(Y_train, Y_predict_train_bo)

    mse_test_bo = mean_squared_error(Y_test, Y_predict_test_bo)
    rmse_test_bo = mean_squared_error(Y_test, Y_predict_test_bo, squared=False)
    r2_test_bo = r2_score(Y_test, Y_predict_test_bo)

    columns.append(random_state_counter)
    random_state_counter += 1

    mse_train_values.append(mse_training_bo)
    rmse_train_values.append(rmse_training_bo)
    r2_train_values.append(r2_training_bo)

    mse_values.append(mse_test_bo)
    rmse_values.append(rmse_test_bo)
    r2_values.append(r2_test_bo)
    best_hypam1_values.append(best_hidden_layer_sizes)
    best_hypam2_values.append(best_alpha)
    best_hypam3_values.append(best_learning_rate)

mean_mse_train = np.mean(mse_train_values)
mean_rmse_train = np.mean(rmse_train_values)
mean_r2_train = np.mean(r2_train_values)
mean_mse = np.mean(mse_values)
mean_rmse = np.mean(rmse_values)
mean_r2 = np.mean(r2_values)
mean_best_hypam1 = np.mean(best_hypam1_values)
mean_best_hypam2 = np.mean(best_hypam2_values)
#mean_best_hypam3 = np.mean(best_hypam3_values)

std_mse_train = np.std(mse_train_values)
std_rmse_train = np.std(rmse_train_values)
std_r2_train = np.std(r2_train_values)
std_mse = np.std(mse_values)
std_rmse = np.std(rmse_values)
std_r2 = np.std(r2_values)
std_best_hypam1 = np.std(best_hypam1_values)
std_best_hypam2 = np.std(best_hypam2_values)
#std_best_hypam3 = np.std(best_hypam3_values)

mse_train_values.append(mean_mse_train)
mse_train_values.append(std_mse_train)
rmse_train_values.append(mean_rmse_train)
rmse_train_values.append(std_rmse_train)
r2_train_values.append(mean_r2_train)
r2_train_values.append(std_r2_train)
mse_values.append(mean_mse)
mse_values.append(std_mse)
rmse_values.append(mean_rmse)
rmse_values.append(std_rmse)
r2_values.append(mean_r2)
r2_values.append(std_r2)
best_hypam1_values.append(mean_best_hypam1)
best_hypam1_values.append(std_best_hypam1)
best_hypam2_values.append(mean_best_hypam2)
best_hypam2_values.append(std_best_hypam2)
best_hypam3_values.append(np.nan)
best_hypam3_values.append(np.nan)


means_mse_train.append(mean_mse_train)
stds_mse_train.append(std_mse_train)
means_rmse_train.append(mean_rmse_train)
stds_rmse_train.append(std_rmse_train)
means_r2_train.append(mean_r2_train)
stds_r2_train.append(std_r2_train)
means_mse.append(mean_mse)
stds_mse.append(std_mse)
means_rmse.append(mean_rmse)
stds_rmse.append(std_rmse)
means_r2.append(mean_r2)
stds_r2.append(std_r2)    

print("average MSE on test set:", mean_mse)
print("Standard Deviation of MSE:", std_mse)
print("average RMSE on test set:", mean_rmse)
print("Standard Deviation of RMSE:", std_rmse)
print("average R2 on test set:", mean_r2)
print("Standard Deviation of R2:", std_r2)

summary = np.array([mse_train_values, rmse_train_values, r2_train_values, mse_values, rmse_values, r2_values,
                    best_hypam1_values, best_hypam2_values, best_hypam3_values])
columns.append("mean")
columns.append("std")
df_summary = pd.DataFrame(summary, columns=columns)
index_values = ['mse_train', 'rmse_train', 'r2_train', 'mse_test', 'rmse_test', 'r2_test',
                hyperparameters[0], hyperparameters[1], hyperparameters[2]]
df_summary = df_summary.set_index(pd.Index(index_values))
print(df_summary)
df_summary.to_csv('analytics_summary_xdnn.csv', index=True)

print(best_hypam3_values)

# linear ridge
random_state_counter_init = 42
random_state_counter = random_state_counter_init
runs = 10

mse_train_values = []
rmse_train_values = []
r2_train_values = []

mse_values = []
rmse_values = []
r2_values = []

best_hypam1_values = []
best_hypam2_values = []
best_hypam3_values = []

hyperparameters = ["alpha"]

columns = []

while random_state_counter < (random_state_counter_init + runs):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state_counter)

    # standardize data
    scaler = StandardScaler() # Initialize the StandardScaler
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # normalize data with min-max scaling
    scaler = MinMaxScaler() # Initialize the MinMaxScaler
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    def objective_function(params):
        ridge_regressor = Ridge(
            alpha=params['alpha'],
            random_state=random_state_counter
        )

        ridge_regressor.fit(X_train, Y_train)
        Y_pred = ridge_regressor.predict(X_train)

        mse = mean_squared_error(Y_train, Y_pred)
        return mse

    param_space = {
        'alpha': Real(1e-3, 1e1)
    }

    opt = BayesSearchCV(
        estimator=Ridge(),
        search_spaces=param_space,
        scoring='neg_mean_squared_error',
        n_iter=60,
        random_state=random_state_counter
    )
    
    #opt.fit(X_train, Y_train.ravel())
    opt.fit(X_train, Y_train)

    best_params = opt.best_params_
    best_alpha = best_params['alpha']

    final_ridge = Ridge(alpha=best_alpha, random_state=random_state_counter)
    final_ridge.fit(X_train, Y_train)

    Y_predict_test_ridge = final_ridge.predict(X_test)
    Y_predict_train_ridge = final_ridge.predict(X_train)

    mse_training_ridge = mean_squared_error(Y_train, Y_predict_train_ridge)
    rmse_training_ridge = mean_squared_error(Y_train, Y_predict_train_ridge, squared=False)
    r2_training_ridge = r2_score(Y_train, Y_predict_train_ridge)

    mse_test_ridge = mean_squared_error(Y_test, Y_predict_test_ridge)
    rmse_test_ridge = mean_squared_error(Y_test, Y_predict_test_ridge, squared=False)
    r2_test_ridge = r2_score(Y_test, Y_predict_test_ridge)

    columns.append(random_state_counter)
    random_state_counter += 1

    mse_train_values.append(mse_training_ridge)
    rmse_train_values.append(rmse_training_ridge)
    r2_train_values.append(r2_training_ridge)

    mse_values.append(mse_test_ridge)
    rmse_values.append(rmse_test_ridge)
    r2_values.append(r2_test_ridge)
    best_hypam1_values.append(best_alpha)
    best_hypam2_values.append(np.nan)
    best_hypam3_values.append(np.nan)

mean_mse_train = np.mean(mse_train_values)
mean_rmse_train = np.mean(rmse_train_values)
mean_r2_train = np.mean(r2_train_values)
mean_mse = np.mean(mse_values)
mean_rmse = np.mean(rmse_values)
mean_r2 = np.mean(r2_values)
mean_best_hypam1 = np.mean(best_hypam1_values)

std_mse_train = np.std(mse_train_values)
std_rmse_train = np.std(rmse_train_values)
std_r2_train = np.std(r2_train_values)
std_mse = np.std(mse_values)
std_rmse = np.std(rmse_values)
std_r2 = np.std(r2_values)
std_best_hypam1 = np.std(best_hypam1_values)

mse_train_values.append(mean_mse_train)
mse_train_values.append(std_mse_train)
rmse_train_values.append(mean_rmse_train)
rmse_train_values.append(std_rmse_train)
r2_train_values.append(mean_r2_train)
r2_train_values.append(std_r2_train)
mse_values.append(mean_mse)
mse_values.append(std_mse)
rmse_values.append(mean_rmse)
rmse_values.append(std_rmse)
r2_values.append(mean_r2)
r2_values.append(std_r2)
best_hypam1_values.append(mean_best_hypam1)
best_hypam1_values.append(std_best_hypam1)
best_hypam2_values.append(np.nan)
best_hypam2_values.append(np.nan)
best_hypam3_values.append(np.nan)
best_hypam3_values.append(np.nan)

means_mse_train.append(mean_mse_train)
stds_mse_train.append(std_mse_train)
means_rmse_train.append(mean_rmse_train)
stds_rmse_train.append(std_rmse_train)
means_r2_train.append(mean_r2_train)
stds_r2_train.append(std_r2_train) 
means_mse.append(mean_mse)
stds_mse.append(std_mse)
means_rmse.append(mean_rmse)
stds_rmse.append(std_rmse)
means_r2.append(mean_r2)
stds_r2.append(std_r2) 

print("average MSE on test set:", mean_mse)
print("Standard Deviation of MSE:", std_mse)
print("average RMSE on test set:", mean_rmse)
print("Standard Deviation of RMSE:", std_rmse)
print("average R2 on test set:", mean_r2)
print("Standard Deviation of R2:", std_r2)

summary = np.array([mse_train_values, rmse_train_values, r2_train_values, mse_values, rmse_values, r2_values,
                    best_hypam1_values])
columns.append("mean")
columns.append("std")
df_summary = pd.DataFrame(summary, columns=columns)
index_values = ['mse_train', 'rmse_train', 'r2_train', 'mse_test', 'rmse_test', 'r2_test',
                hyperparameters[0]]
df_summary = df_summary.set_index(pd.Index(index_values))
print(df_summary)
df_summary.to_csv('analytics_summary_ridge.csv', index=True)



# knn
random_state_counter_init = 42
random_state_counter = random_state_counter_init
runs = 10

mse_train_values = []
rmse_train_values = []
r2_train_values = []

mse_values = []
rmse_values = []
r2_values = []

best_hypam1_values = []
best_hypam2_values = []
best_hypam3_values = []

hyperparameters = ["n_neighbors", "weights", "p"]

columns = []

while random_state_counter < (random_state_counter_init + runs):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state_counter)

    # standardize data
    scaler = StandardScaler() # Initialize the StandardScaler
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # normalize data with min-max scaling
    scaler = MinMaxScaler() # Initialize the MinMaxScaler
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    def objective_function(params):
        knn_regressor = KNeighborsRegressor(
            n_neighbors=params['n_neighbors'],
            weights=params['weights'],
            p=params['p']
        )

        knn_regressor.fit(X_train, Y_train)
        Y_pred = knn_regressor.predict(X_train)

        mse = mean_squared_error(Y_train, Y_pred)
        return mse

    param_space = {
        'n_neighbors': Integer(1, 10),
        'weights': Categorical(['uniform', 'distance']),
        'p': Integer(1, 2)
    }

    opt = BayesSearchCV(
        estimator=KNeighborsRegressor(),
        search_spaces=param_space,
        scoring='neg_mean_squared_error',
        n_iter=60,
        random_state=random_state_counter
    )
    
    #opt.fit(X_train, Y_train.ravel())
    opt.fit(X_train, Y_train)

    best_params = opt.best_params_
    best_n_neighbors = best_params['n_neighbors']
    best_weights = best_params['weights']
    best_p = best_params['p']

    final_knn = KNeighborsRegressor(**best_params)
    final_knn.fit(X_train, Y_train)

    Y_predict_test_knn = final_knn.predict(X_test)
    Y_predict_train_knn = final_knn.predict(X_train)

    mse_training_knn = mean_squared_error(Y_train, Y_predict_train_knn)
    rmse_training_knn = mean_squared_error(Y_train, Y_predict_train_knn, squared=False)
    r2_training_knn = r2_score(Y_train, Y_predict_train_knn)

    mse_test_knn = mean_squared_error(Y_test, Y_predict_test_knn)
    rmse_test_knn = mean_squared_error(Y_test, Y_predict_test_knn, squared=False)
    r2_test_knn = r2_score(Y_test, Y_predict_test_knn)

    columns.append(random_state_counter)
    random_state_counter += 1

    mse_train_values.append(mse_training_knn)
    rmse_train_values.append(rmse_training_knn)
    r2_train_values.append(r2_training_knn)

    mse_values.append(mse_test_knn)
    rmse_values.append(rmse_test_knn)
    r2_values.append(r2_test_knn)
    best_hypam1_values.append(best_n_neighbors)
    best_hypam2_values.append(best_weights)
    best_hypam3_values.append(best_p)

mean_mse_train = np.mean(mse_train_values)
mean_rmse_train = np.mean(rmse_train_values)
mean_r2_train = np.mean(r2_train_values)
mean_mse = np.mean(mse_values)
mean_rmse = np.mean(rmse_values)
mean_r2 = np.mean(r2_values)
mean_best_hypam1 = np.mean(best_hypam1_values)

std_mse_train = np.std(mse_train_values)
std_rmse_train = np.std(rmse_train_values)
std_r2_train = np.std(r2_train_values)
std_mse = np.std(mse_values)
std_rmse = np.std(rmse_values)
std_r2 = np.std(r2_values)
std_best_hypam1 = np.std(best_hypam1_values)

mse_train_values.append(mean_mse_train)
mse_train_values.append(std_mse_train)
rmse_train_values.append(mean_rmse_train)
rmse_train_values.append(std_rmse_train)
r2_train_values.append(mean_r2_train)
r2_train_values.append(std_r2_train)
mse_values.append(mean_mse)
mse_values.append(std_mse)
rmse_values.append(mean_rmse)
rmse_values.append(std_rmse)
r2_values.append(mean_r2)
r2_values.append(std_r2)
best_hypam1_values.append(mean_best_hypam1)
best_hypam1_values.append(std_best_hypam1)
best_hypam2_values.append(np.nan)
best_hypam2_values.append(np.nan)
best_hypam3_values.append(np.nan)
best_hypam3_values.append(np.nan)

means_mse_train.append(mean_mse_train)
stds_mse_train.append(std_mse_train)
means_rmse_train.append(mean_rmse_train)
stds_rmse_train.append(std_rmse_train)
means_r2_train.append(mean_r2_train)
stds_r2_train.append(std_r2_train) 
means_mse.append(mean_mse)
stds_mse.append(std_mse)
means_rmse.append(mean_rmse)
stds_rmse.append(std_rmse)
means_r2.append(mean_r2)
stds_r2.append(std_r2) 

print("average MSE on test set:", mean_mse)
print("Standard Deviation of MSE:", std_mse)
print("average RMSE on test set:", mean_rmse)
print("Standard Deviation of RMSE:", std_rmse)
print("average R2 on test set:", mean_r2)
print("Standard Deviation of R2:", std_r2)

summary = np.array([mse_train_values, rmse_train_values, r2_train_values, mse_values, rmse_values, r2_values,
                    best_hypam1_values, best_hypam2_values, best_hypam3_values])
columns.append("mean")
columns.append("std")
df_summary = pd.DataFrame(summary, columns=columns)
index_values = ['mse_train', 'rmse_train', 'r2_train', 'mse_test', 'rmse_test', 'r2_test',
                hyperparameters[0], hyperparameters[1], hyperparameters[2]]
df_summary = df_summary.set_index(pd.Index(index_values))
print(df_summary)
df_summary.to_csv('analytics_summary_knn.csv', index=True)









summary_all = np.array([means_mse_train, stds_mse_train, means_rmse_train, stds_rmse_train, means_r2_train, stds_r2_train, means_mse, stds_mse, means_rmse, stds_rmse, means_r2, stds_r2])

columns = ["gpr", "krr", "krr_linear", "rf", "dt", "lr", "xDNN", "linear ridge", "knn"]

df_summary_all = pd.DataFrame(summary_all, columns=columns)
index_values = ['means_mse_train', 'stds_mse_train', 'means_rmse_train', 'stds_rmse_train', 'means_r2_train', 'stds_r2_train', 'means_mse_test', 'stds_mse_test', 'means_rmse_test', 'stds_rmse_test', 'means_r2_test', 'stds_r2_test']
df_summary_all = df_summary_all.set_index(pd.Index(index_values))
print(df_summary)
df_summary_all.to_csv('analytics_summary_all.csv', index=True)

summary_all = np.array([means_mse_train, stds_mse_train, means_rmse_train, stds_rmse_train, means_r2_train, stds_r2_train, means_mse, stds_mse, means_rmse, stds_rmse, means_r2, stds_r2])
print(summary_all)

print(means_mse_train)