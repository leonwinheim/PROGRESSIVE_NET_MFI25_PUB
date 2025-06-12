####################################################################################################
# Description: This script is used to compare the classification results of the different models.
# The classification task is the moon dataset
# We evaluate some static evaluation and try out how fast the models can learn the dataset
# Author: Leon Winheim
####################################################################################################import torch
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from models.PROGRESSIVE_NET import PROGRESSIVE
#from models.KBNN import Bayesian_Network_torch as KBNN
import time
import os
import pandas as pd

#******Control Variables******
torch.manual_seed(41)
np.random.seed(41)

run_static = False
run_sequential = True

# Path to save the data
path ="temp/benchmark_moons_static/"

if not os.path.exists(path):
    os.makedirs(path)

#******Data Generation******
training_size = 1500
data_noise = 0.15
X, y = make_moons(n_samples=training_size, noise=data_noise, random_state=0)

#******Model parameters******
# Define the parameters for my one hidden layer network (Relu Activated)
layers = [2,10,10,1]
act_func = ["relu","relu","sigmoid"]

#******Run environment for single bing run******
if run_static:
    number_runs = 10

    accuracy_list = []

    for i in range(number_runs):
        # Shuffle Data
        shuffle_idx = torch.randperm(X.shape[0])
        X = X[shuffle_idx]
        y = y[shuffle_idx]

        # Train-Test Split
        split = int(0.9*training_size)
        x_train = X[:split]
        y_train = y[:split]
        x_test = X[split:]
        y_test = y[split:]

        #Convert to torch (training data)
        x_train = torch.tensor(x_train,dtype=torch.float32)
        y_train = torch.tensor(y_train,dtype=torch.float32)

        x_test = torch.tensor(x_test,dtype=torch.float32)
        y_test = torch.tensor(y_test,dtype=torch.float32)

        # Generate artificial prediction data
        x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
        y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 40), np.linspace(y_min, y_max, 40))

        x_ext = np.c_[xx.ravel(), yy.ravel()]
        x_ext = torch.tensor(x_ext, dtype=torch.float32)

        # Make Model
        num_particles = 25000
        algo = "progressive_lcd"
        train_mode = "sequential"
        model = PROGRESSIVE(layers, act_func,num_particles,algo,train_mode,meas_variance=0.1)   
        #Results are superior for a "bigger" noise value than the variance that is given to mons

        # Train Model
        start = time.time()
        model.train(x_train, y_train,train_mode)
        end = time.time()

        filename = path+"Progressive_NP"+str(num_particles)+"_Mode"+algo+"_train_"+train_mode+"_L"+str(data_noise)+".pkl"

        model.save_particles(filename)

        # Predict
        y_pred = model.predict(x_test)

        # Compute output
        mean_pred = y_pred.mean(dim=1)
        var_pred = y_pred.var(dim=1)

        # Compute accuracy
        accuracy = (mean_pred.round() == y_test).float().mean().item()
        accuracy_list.append(accuracy)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Time taken: {end - start:.4f} seconds")

    print(f"Results from all runs: Mean Accuracy: {np.mean(accuracy_list):.4f}, Std Accuracy: {np.std(accuracy_list):.4f}")

    # Graphics (From the last model)
    pred_ext = model.predict(x_ext)
    mean_ext = pred_ext.mean(dim=1)
    var_ext = pred_ext.var(dim=1)
    mean_ext = mean_ext.reshape(xx.shape)
    var_ext = var_ext.reshape(xx.shape)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })

    plt.figure(figsize=(6, 2.8))

    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, mean_ext, levels=50, cmap='RdBu', alpha=0.8)
    # Plot only 20% of available points
    n_train = int(0.2 * x_train.shape[0])
    n_test = int(0.2 * x_test.shape[0])
    idx_train = np.random.choice(x_train.shape[0], n_train, replace=False)
    idx_test = np.random.choice(x_test.shape[0], n_test, replace=False)
    plt.scatter(x_train[idx_train, 0], x_train[idx_train, 1], c=y_train[idx_train], s=20, cmap='RdBu', edgecolor='k')
    plt.scatter(x_test[idx_test, 0], x_test[idx_test, 1], c=y_test[idx_test], s=50, cmap='RdBu', edgecolor='k', marker='x')
    plt.title('Classification')
    plt.colorbar()


    plt.subplot(1, 2, 2)
    plt.contourf(xx, yy, var_ext, levels=50, cmap='RdBu', alpha=0.8)
    plt.scatter(x_train[idx_train, 0], x_train[idx_train, 1], c=y_train[idx_train], s=20, cmap='RdBu', edgecolor='k')
    plt.title('Classification uncertainty')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(path + "moons.pdf",bbox_inches='tight')
    plt.savefig(path + "moons.svg",bbox_inches='tight')

#******Run environment for sequential tests******
if run_sequential:
    number_runs = 10

    test_sizes = [5,50,100,250,500,1000,1350]   #Define the intermediate testing scenarios

    accuracy_list = []
    accuracy_list_kbnn = []

    for i in range(number_runs):
        print(f"Run {i+1}/{number_runs}")
        # Shuffle Data
        shuffle_idx = torch.randperm(X.shape[0])
        X = X[shuffle_idx]
        y = y[shuffle_idx]

        # Train-Test Split
        split = int(0.9*training_size)
        x_train = X[:split]
        y_train = y[:split]
        x_test = X[split:]
        y_test = y[split:]

        #Convert to torch (training data)
        x_train = torch.tensor(x_train,dtype=torch.float32)
        y_train = torch.tensor(y_train,dtype=torch.float32)

        x_test = torch.tensor(x_test,dtype=torch.float32)
        y_test = torch.tensor(y_test,dtype=torch.float32)

        # Generate artificial prediction data
        x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
        y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 40), np.linspace(y_min, y_max, 40))

        x_ext = np.c_[xx.ravel(), yy.ravel()]
        x_ext = torch.tensor(x_ext, dtype=torch.float32)

        # Make Model
        num_particles = 25000
        algo = "progressive_lcd"
        train_mode = "sequential"
        model = PROGRESSIVE(layers, act_func,num_particles,algo,train_mode,meas_variance=0.1)
        #Results are superior for a "bigger" noise value than the variance that is given to moons

        # Make KBNN
        #model_kbnnn = KBNN(layers,act_func,verbose=False, noise=0.01, prior_noise=1.0,use_cuda=False)
        
        num_before = 0
        for num in test_sizes:
            # Take the partial data out
            x_train_part = x_train[num_before:num]
            y_train_part = y_train[num_before:num]
            num_before = num

            # Train Model (Progressive)
            start = time.time()
            model.train(x_train_part, y_train_part,train_mode)
            end = time.time()

            #Train Model (KBNN)
            #start_kbnn = time.time()
            #model_kbnnn.train(x_train_part, y_train_part.unsqueeze(-1))
            #end_kbnn = time.time()

            # Predict (Progressive)
            y_pred = model.predict(x_test)

            # Compute output
            mean_pred = y_pred.mean(dim=1)
            var_pred = y_pred.var(dim=1)

            # Predict (KBNN)
            #y_pred_kbnn,var_pred_kbnn = model_kbnnn.predict(x_test)
            #y_pred_kbnn = torch.tensor(y_pred_kbnn).squeeze()
            #var_pred_kbnn = torch.tensor(var_pred_kbnn).squeeze()

            # Compute accuracy (Progressive)
            accuracy = (mean_pred.round() == y_test).float().mean().item()
            accuracy_list.append([num,accuracy,end-start])
            print(f"Test size: {num}, Accuracy: {accuracy:.4f},Train time: {end - start:.4f} s")

            # Compute accuracy (KBNN)
            #accuracy_kbnn = (y_pred_kbnn.round() == y_test).float().mean().item()
            #accuracy_list_kbnn.append([num,accuracy_kbnn,end_kbnn - start_kbnn])
            #print(f"Test size: {num}, Accuracy KBNN: {accuracy_kbnn:.4f},Train time: {end_kbnn - start_kbnn:.4f} s")
    
    path_ext = path+"result/"
    if not os.path.exists(path_ext):
        os.makedirs(path_ext)

    if len(accuracy_list) == 0:
        accuracy_list = pd.read_csv(path_ext + "Progressive_Accuracy.csv").values.tolist()
        #accuracy_list_kbnn = pd.read_csv(path_ext + "KBNN_Accuracy.csv").values.tolist()

    #Evaluate
    accuracy_list = np.array(accuracy_list)
    accuracy_list = pd.DataFrame(accuracy_list, columns=["Test Size","Accuracy","Time"])
    mean_accuracy = accuracy_list.groupby("Test Size")["Accuracy"].mean()
    std_accuracy = accuracy_list.groupby("Test Size")["Accuracy"].std()
    mean_time = accuracy_list.groupby("Test Size")["Time"].mean()
    std_time = accuracy_list.groupby("Test Size")["Time"].std()

    #accuracy_list_kbnn = np.array(accuracy_list_kbnn)
    #accuracy_list_kbnn = pd.DataFrame(accuracy_list_kbnn, columns=["Test Size","Accuracy","Time"])
    #mean_accuracy_kbnn = accuracy_list_kbnn.groupby("Test Size")["Accuracy"].mean()
    #std_accuracy_kbnn = accuracy_list_kbnn.groupby("Test Size")["Accuracy"].std()
    #mean_time_kbnn = accuracy_list_kbnn.groupby("Test Size")["Time"].mean()
    #std_time_kbnn = accuracy_list_kbnn.groupby("Test Size")["Time"].std()

    #print("Mean Accuracy")
    #for i in mean_accuracy.index:
        #print(f"Progressive: {mean_accuracy[i]:4f}, KBNN: {mean_accuracy_kbnn[i]:4f}")

    # Save the results
    accuracy_list.to_csv(path_ext+"Progressive_Accuracy.csv",index=False)
    #accuracy_list_kbnn.to_csv(path_ext+"KBNN_Accuracy.csv",index=False)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })

    plt.rcParams['lines.linewidth'] = 2

    #Make graphics
    factor = 1.6
    plt.figure(figsize=(factor*3.45, factor*1.8))
    plt.errorbar(
        mean_accuracy.index,
        100*mean_accuracy.values,
        yerr=100*std_accuracy.values,
        label="Progressive",
        marker='o',
        capsize=5,
        alpha=0.8,
        color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0]  # Use the default tab:blue
    )
    # plt.errorbar(
    #     mean_accuracy_kbnn.index,
    #     100*mean_accuracy_kbnn.values,
    #     yerr=100*std_accuracy_kbnn.values,
    #     label="KBNN",
    #     marker='o',
    #     capsize=5,
    #     alpha=0.8,
    #     color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1]  # Use the default tab:orange
    # )
    plt.xlabel("Training points")
    plt.ylabel("Accuracy in \%")
    plt.legend()
    plt.title("Accuracy vs. number of training points for the Moon Dataset")
    plt.tight_layout()
    plt.savefig(path_ext + "Accuracy_Comparison.pdf",bbox_inches='tight')
    plt.savefig(path_ext + "Accuracy_Comparison.svg",bbox_inches='tight')
