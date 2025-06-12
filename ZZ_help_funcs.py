"""
ZZ_help_funcs.py

This module contains some utilitiy functions.

"""
import os, io, time, requests, zipfile
import numpy as np
from sklearn.datasets import *
from sklearn import preprocessing
import pandas as pd
import torch
import pickle 

def create_data(dataset, data_size, feature_range):
    """Different kinds of dataset and tasks"""
    # Binary Classification Dataset
    if dataset == "dataset_moon":
        X, Y = make_moons(n_samples=data_size, noise=0.15, random_state=0)

    elif dataset == "dataset_circles":
        X, Y = make_circles(n_samples=data_size, noise=0.01, random_state=0)

    elif dataset == "dataset_synthetic_regression":
        # X_1 = np.random.uniform(-4, -2, size=int(0.5 * data_size))
        # X_2 = np.random.uniform(-2, 2, size=int(0 * data_size))
        # X_3 = np.random.uniform(2, 4, size=int(0.5 * data_size))
        # X = np.concatenate((X_1, X_2, X_3))
        X = np.random.uniform(-4, 4, size=data_size)
        Y = np.power(X, 3) + np.random.normal(0, 3, size=data_size)
        # Y = Y.reshape(-1, 1)
        X = X.reshape(-1, 1)

    elif dataset == "dataset_boston":
        raise NotImplementedError("Boston dataset is canceled")
        #X, Y = load_boston(return_X_y=True)

    elif dataset == "dataset_concrete":
        df = pd.read_excel(
            'http://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls')
        X = df.drop(df.columns[-1], axis=1).to_numpy()
        Y = df[df.columns[-1]].to_numpy()

    elif dataset == "dataset_energy":
        df = pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx')
        X = df.drop(df.columns[[-1, -2]], axis=1).to_numpy()
        Y = df[df.columns[-1]].to_numpy()

    elif dataset == "dataset_wine_quality":
        df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',
                         sep=';')
        X = df.drop('quality', axis=1).to_numpy()
        Y = df['quality'].to_numpy()

    elif dataset == "dataset_naval_propulsion":
        zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI CBM Dataset.zip'
        r = requests.get(zip_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        df = pd.read_csv(z.open('UCI CBM Dataset/data.txt'), sep='  ', header=None)
        X = df.iloc[:, :16].to_numpy()
        Y = df.iloc[:, 16].to_numpy()

    elif dataset == "dataset_yacht":
        df = pd.read_fwf('https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data').dropna()
        X = df.iloc[:, :6].to_numpy()
        Y = df.iloc[:, 6].to_numpy()

    elif dataset == "dataset_kin8nm":
        url = 'https://www.openml.org/data/get_csv/3626/dataset_2175_kin8nm.arff'
        data = pd.read_csv(url).values
        X = data[:, :-1]
        Y = data[:, -1]

    elif dataset == "dataset_protein":
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv'

        data = pd.read_csv(url).values
        data = np.concatenate([data[:, 1:], data[:, 0, None]], 1)
        X = data[:, :-1]
        Y = data[:, -1]

    elif dataset == "dataset_power":
        from urllib.request import urlopen
        from zipfile import ZipFile
        from io import BytesIO
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip'
        with urlopen(url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall('/tmp/')

        data = pd.read_excel('/tmp/CCPP//Folds5x2_pp.xlsx').values
        X = data[:, :-1]
        Y = data[:, -1]

    elif dataset == "dataset_year":
        from urllib.request import urlopen
        from zipfile import ZipFile
        from io import BytesIO
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip'
        with urlopen(url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall('/tmp/')

        data = pd.read_csv('/tmp/YearPredictionMSD.txt', delimiter=',', header=None)
        cols = data.columns.tolist()
        cols = cols[1:] + [cols[0]]
        data = data[cols].values
        data = data[:463810, ]
        X = data[:, :-1]
        Y = data[:, -1]

    x_scaler = None
    y_scaler = None
    
    if not dataset == "dataset_synthetic_regression":
        # Chose Scaler
        x_scaler = preprocessing.MinMaxScaler(feature_range=(feature_range[0], feature_range[1]))
        # x_scaler = preprocessing.StandardScaler()

        #Scale
        X = x_scaler.fit_transform(X)

        # Decide wether output should be scaled (normally not I guess?)
        output_scale = False
        if output_scale:
            Y_mean = np.mean(Y)
            Y_std = np.std(Y)
            Y = (Y - Y_mean) / Y_std
            y_scaler = [Y_mean, Y_std]
    else:
        print("Nothing is scaled!!!")

    return X, Y, x_scaler, y_scaler

def save_uci_data(dataset):
    feature_range = [-1, 1]

    if dataset == "dataset_boston":
        print("Canceled")
    else:
        print("Save dataset:", dataset)
        X, Y, x_scaler, y_scaler = create_data(dataset=dataset, data_size=None, feature_range=feature_range)

        # Save everything in a dictionary in one file per dataset
        data_dict = {"name": dataset,
            "X": X,
            "Y": Y,
            "x_scaler": x_scaler,
            "y_scaler": y_scaler
        }
        output_dir = 'datasets/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, f"{dataset}_data.pkl"), "wb") as f:
            pickle.dump(data_dict, f)

def compute_uce(data:torch.tensor,M,n):
    """
    Function to compute the uncertainty calibration error (UCE).
    Data has 4 columns: [x_test, y_test, mean_pred, var_pred]
    M is the number of bins to use.
    n is the number of samples in the dataset.
    """
    assert data.shape == (n,4)
    
    # Find mimimal and maximal values in the first column
    min_val = torch.min(data[:,0])
    max_val = torch.max(data[:,0])

    # Compute bin width
    bin_width = (max_val - min_val) / M

    # Indice sets for every bin
    indices = []
    for i in range(M):
        indices.append(torch.where((data[:,0] >= min_val + i*bin_width) & (data[:,0] < min_val + (i+1)*bin_width))[0].tolist())

    #print(f"Generated {len(indices)} bins! (Wanted {M})")

    #Initialize the UCE
    uce = 0
    for B_m in indices:
        # Compute pre-factor
        factor = len(B_m)/n
        #Compute the  err(Bm) value
        err_B_m = 0
        for i in B_m:
            # Test minus Prediction mean squared
            err_B_m += (data[i,1] - data[i,2])**2
        # Mean squared error
        err_B_m = err_B_m / len(B_m)
        #Compute the uncertainty value
        unc_B_m = 0
        for i in B_m:
            #Variance of the prediction
            unc_B_m += data[i,3]
        # Mean value
        unc_B_m = unc_B_m / len(B_m)
        #Add to UCE
        uce += factor*abs(err_B_m - unc_B_m)

    return uce

def compute_nll(mean_pred: torch.Tensor, var_pred: torch.Tensor, y_test: torch.Tensor):
    """
    Vectorized negative log-likelihood for Gaussian predictions.
    """
    nll = 0.5 * torch.log(2 * torch.pi * var_pred) + ((y_test - mean_pred) ** 2) / (2 * var_pred)
    
    return nll.sum()/ len(y_test)

def evaluate(path,file,sort_test_data=False,bins_uce=3):
    """
    Evaluates the result files from the regression tasks.
    Reads Pickle-Files and computes the RMSE, UCE, NLL and returns a dictionary with the results.
    """
    #Load the data from the pickle file
    with open(os.path.join(path, file), "rb") as f:
        data = pickle.load(f)

    # Start evaluation
    #run_name = data['run_name']            #name out of parameterset
    run_name = os.path.splitext(file)[0]    #name out of filename

    #Sort the test data and predictions
    x_test = data["x_test"]
    y_test = data["y_test"]
    mean_pred = data["mean_pred"]
    var_pred = data["var_pred"]

    if sort_test_data:
        # Sort indices by ascending order according to x_test
        sorted_indices = torch.argsort(x_test, dim=0)
        x_test = x_test[sorted_indices.flatten()]
        y_test = y_test[sorted_indices.flatten()]

        x_test = x_test.reshape(-1,1)

        mean_pred = mean_pred[sorted_indices.flatten()]
        var_pred = var_pred[sorted_indices.flatten()]

    y_test = y_test.reshape(-1,1)
    mean_pred = mean_pred.reshape(-1,1)
    var_pred = var_pred.reshape(-1,1)

    # Compute RMSE
    rmse = torch.sqrt(torch.mean(((y_test.reshape(-1,1)) - mean_pred.reshape(-1,1))**2))

    # Compute the UCE
    if len(x_test.shape)==1 or x_test.shape[1] == 1:
        #Assemble input tensor (stacks the columns side by side, column one is x_test, column two is y_test, column three is mean_pred, column four is var_pred)
        data_uce = torch.cat((x_test.reshape(-1,1),y_test.reshape(-1,1),mean_pred.reshape(-1,1),var_pred.reshape(-1,1)),dim=1)
        #Compute UCE
        uce = compute_uce(data_uce,bins_uce,mean_pred.shape[0])
    else:
        uce = torch.tensor(0.0)

    # Compute the NLL
    nll = compute_nll(mean_pred, var_pred, y_test)

    # Extract training and prediction times
    train_time = data["train_time"]
    pred_time = data["pred_time"]

    results = {
        "run_name": run_name,
        "rmse": rmse,
        "uce": uce,
        "train_time": train_time,
        "pred_time": pred_time,
        "mean_pred": mean_pred,
        "var_pred": var_pred,
        "nll": nll,
        "x_test": x_test,
        "y_test": y_test,
    }

    return results