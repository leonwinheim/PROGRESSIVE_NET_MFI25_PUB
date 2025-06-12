####################################################################################################
# Description: This script is used to compare the classification results of the different models.
# The classification task is the moon dataset, and it is rotated
# Author: Leon Winheim
# Date: 23.04.2025
####################################################################################################import torch
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from models.PROGRESSIVE_NET import PROGRESSIVE
import time
import os

#******Control Variables******
torch.manual_seed(41)
np.random.seed(41)

# Path to save the data
path ="temp/benchmark_moons_online/"

if not os.path.exists(path):
    os.makedirs(path)

number_runs = 5
for j in range(number_runs):
    path = f"temp/benchmark_moons_online/{j}/"
    if not os.path.exists(path):
        os.makedirs(path)
    #******Data Generation******
    training_size = 1500
    new_training_size = 100
    data_noise = 0.15
    X, y = make_moons(n_samples=training_size, noise=data_noise, random_state=42)
    X -= X.mean(axis=0)  # Center the moon data

    # Scale the moon data to [-1, 1] for both directions
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X = 2 * (X - X_min) / (X_max - X_min) - 1


    #******Model parameters******
    # Define the parameters for my one hidden layer network (Relu Activated)
    layers = [2,10,10,1]
    act_func = ["relu","relu","sigmoid"]

    #******Run environment******
    # Do the big static training
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
    model = PROGRESSIVE(layers, act_func,num_particles,algo,train_mode,meas_variance= 0.1 ,prior_variance=1.0)
    #Results are superior for a "bigger" noise value than the variance that is given to mons

    # Train Model
    start = time.time()
    model.train(x_train, y_train)
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

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Time taken: {end - start:.4f} seconds")

    # Do the rotation and train again
    angle_increment = 20.0 #Degrees!
    angle_increment = torch.tensor(np.deg2rad(angle_increment)) #Convert to radians
    rotations = 10 #Number of rotations

    for i in range(1, rotations):
        angle = i*angle_increment

        # Rotate every 2D point in X with the angle_increment
        rotation_matrix = torch.tensor([
            [torch.cos(angle), -torch.sin(angle)],
            [torch.sin(angle), torch.cos(angle)]
        ], dtype=torch.float32)

        X_new = torch.matmul(torch.tensor(X, dtype=torch.float32), rotation_matrix.T)

        # Select 100 random indices from X_new
        random_indices = torch.randperm(X_new.shape[0])[:new_training_size]
        X_partial = X_new[random_indices]
        y_partial = y[random_indices]
        X_partial = torch.tensor(X_partial, dtype=torch.float32)
        y_partial = torch.tensor(y_partial, dtype=torch.float32)

        # Draw 100 again for testing
        random_indices = torch.randperm(X_new.shape[0])[:new_training_size]
        x_test = X_new[random_indices]
        y_test = y[random_indices]
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        # Train the pretrained model on the rotated data
        model.train(X_partial, y_partial, train_mode)

        filename = path+"Progressive_NP"+str(num_particles)+"_Mode"+algo+"_train_"+train_mode+"_L"+str(data_noise)+"_angle_"+str(i*angle_increment)+".pkl"
        model.save_particles(filename)

        # Predict
        y_pred = model.predict(x_test)

        # Compute Accuracy
        mean_pred = y_pred.mean(dim=1)
        var_pred = y_pred.var(dim=1)
        accuracy = (mean_pred.round() == y_test).float().mean().item()
        print(f"Rotation {i}: Accuracy: {accuracy:.4f}")
        print(f"Time taken: {end - start:.4f} seconds")

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
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.contourf(xx, yy, mean_ext, levels=50, cmap='RdBu', alpha=0.8)
        plt.scatter(X_partial[:, 0], X_partial[:, 1], c=y_partial, s=10, cmap='RdBu', edgecolor='k')
        plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, s=10, cmap='RdBu', edgecolor='k', marker='x')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(r'\textbf{Mean Prediction} (Acc: %.4f)' % accuracy)
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.contourf(xx, yy, var_ext, levels=50, cmap='RdBu', alpha=0.8)
        plt.scatter(X_new[:, 0], X_new[:, 1], c=y, s=10, cmap='RdBu', edgecolor='k')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(r'\textbf{Variance Prediction}')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(path+"Progressive_NP"+str(num_particles)+"_Mode"+algo+"_train_"+train_mode+"_L"+str(data_noise)+"run_"+str(i)+".pdf")
        plt.savefig(path+"Progressive_NP"+str(num_particles)+"_Mode"+algo+"_train_"+train_mode+"_L"+str(data_noise)+"run_"+str(i)+".svg")

