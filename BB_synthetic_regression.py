#############################################################################
# Regression benchmark for cubic scalar function f(x) = x^3
# Compare PROGRESSIVE, KBNN and MCMC
# Finalized version with evaluation and graphics
# Author: Leon Winheim
#############################################################################

import torch
import numpy as np
import os
from sklearn import preprocessing
import ZZ_run_funcs as rf
import ZZ_help_funcs as hf
import pickle 
import matplotlib.pyplot as plt

#******Control Variables******
torch.manual_seed(41)
np.random.seed(41)

train_flag = False
evaluate_flag = True

dtype_man = torch.float32
torch.set_default_dtype(dtype_man)

# Path to save the data
path="temp/synth_reg_cubic/100_pts/"

if not os.path.exists(path):
    os.makedirs(path)

#******Data Generation******
# Variance of the noisy data
data_noise = 9.0

# Range definition for training
num_points = 100
border = 4

# Noiseless model
def true_model(x):
    return x**3

# Noisy model
def noisy_model(x,noise):
    noise = torch.sqrt(torch.tensor(noise)) * torch.randn(x.size())
    return true_model(x) + noise 

# Define training range
x = torch.linspace(-border,border,num_points)
x_ext = torch.linspace(-1.5*border,1.5*border,num_points)

# Generate Data
y_gt = true_model(x)                # Ground Truth
y_gt_ext = true_model(x_ext)        # Ground Truth extended
y = noisy_model(x,data_noise)       # Noisy, for Training /Testing

# Normalize input data to be between -1 and 1
x_scaler  = preprocessing.MinMaxScaler(feature_range=(-1,1))
x = torch.tensor(x_scaler.fit_transform(x.unsqueeze(-1)),dtype=dtype_man).squeeze()
x_ext = torch.tensor(x_scaler.transform(x_ext.unsqueeze(-1)), dtype=dtype_man).squeeze()

if train_flag:
    #******Run environment******
    # Take care when running this somewhat out of order or with breaks. It is important that for each run
    # the random permutations are the same. This should be secured by the following code.
    # It must be like that because i use RNGs in the Progressive and KBNN classes in between
    number_runs = 5

    # Save specific permutations of the training data
    perm_list = []
    for i in range(number_runs):
        perm = torch.randperm(num_points)
        perm_list.append(perm)

    # Do the runs on the training data
    for i in range(number_runs):

        #******Data Shuffling******
        x = x[perm_list[i]]
        y = y[perm_list[i]]

        #******Train/Test Split******
        split_ratio = 0.8   
        x_train = x[:int(split_ratio*num_points)]
        y_train = y[:int(split_ratio*num_points)]
        x_test = x[int(split_ratio*num_points):]
        y_test = y[int(split_ratio*num_points):]

        #******Architecture Parameters******
        layers = [1,100,1]
        act_func = ["relu","linear"]

        #******Progressive run******
        algo  = "progressive_lcd"
        training_type = "sequential"
        num_particles = 25000

        data_noise_progressive = 9.0

        artificial_noise_progressive = 0.0    # This parameter is only used when using random samples

        prior_noise_progressive = 1.0

        #Assemble parameter tuple
        progressive_pars = (x_train,y_train,x_test,y_test,layers,act_func,algo,training_type,num_particles,
                        data_noise_progressive,artificial_noise_progressive,prior_noise_progressive,i)

        # Check wether run is already made
        parset = progressive_pars
        if data_noise_progressive is None:
            run_name = f"Progressive_NP{parset[8]}_{parset[6]}_{parset[7]}_L{'_LEARNED_'}_A{parset[10]:.4f}_P{parset[11]:.4f}_idx{i}"
        else:
            run_name = f"Progressive_NP{parset[8]}_{parset[6]}_{parset[7]}_L{parset[9]:.4f}_A{parset[10]:.4f}_P{parset[11]:.4f}_idx{i}"

        if os.path.exists(path + run_name + '.pkl'):
            print(f"Run {run_name} already exists. Skipping...")
            result = None
        else:
            # Run
            result = rf.run_PROGRESSIVE(*parset,x_ext = x_ext, y_ext = y_gt_ext)
        if not result is None:
            with open(path+result["run_name"]+'.pkl', 'wb') as f:
                pickle.dump(result, f)
        else:
            print(f"Progressive gave None-Result!")

        #******KBNN run******
        artificial_noise_KBNN = 0.0001

        prior_noise_KBNN = 1.0

        # Create parameter tuple
        kbnn_pars = (x_train,y_train,x_test,y_test,layers,act_func,
                                artificial_noise_KBNN,prior_noise_KBNN,i)

        # Check wether run is already made
        parset = kbnn_pars
        run_name = f"KBNN_A{parset[6]:.6f}_P{parset[7]:.4f}_idx{i}"
        if os.path.exists(path + run_name + '.pkl'):
            print(f"Run {run_name} already exists. Skipping...")
            result = None
        else:
            # Run
            result = rf.run_KBNN(*parset,x_ext = x_ext, y_ext = y_gt_ext)
        if not result is None:
            with open(path+result["run_name"]+'.pkl', 'wb') as f:
                pickle.dump(result, f)
        else:
            print(f"KBNN gave None-Result!")

        #******KBNN run******
        num_samples = 500

        data_noise_MCMC = 9.0

        prior_noise_MCMC = 1.0

        # Create parameter tuple
        mcmc_pars = (x_train,y_train,x_test,y_test,layers[1],
                    prior_noise_MCMC,num_samples,data_noise_MCMC,i)
        
        # Check wether run is already made
        parset = mcmc_pars
        run_name = f"MCMC_NS{parset[6]}_L{parset[7]:.4f}_P{parset[5]}_idx{i}"
        if os.path.exists(path + run_name + '.pkl') :
            print(f"Run {run_name} already exists. Skipping...")
            result = None
        else:
            # Run
            result = rf.run_MCMC(*parset,x_ext = x_ext, y_ext = y_gt_ext)
        if not result is None:
            with open(path+result["run_name"]+'.pkl', 'wb') as f:
                pickle.dump(result, f)
        else:
            print(f"MCMC gave None-Result!")
         
if evaluate_flag:
    #******Evaluation******
    #Generate a list of pkl files in the directory  
    files = os.listdir(path)
    files = [f for f in files if (f.endswith(".pkl"))]
    #remove the combined performance file
    files = [f for f in files if (f != "combined_performance.pkl")]
    files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(path, x)))

    if files == []:
        print("No files found. Do some training first!")
        exit()

    #Evaluate and save
    results = []
    for file in files:
        results.append(hf.evaluate(path,file,sort_test_data=True))

    print(f"Found {len(results)} files!")

    # Combine results with the same name but different index
    # First step: In the list of results, find unique run names up to _idx
    run_names = []
    for result in results:
        run_name = result["run_name"].split("_idx")[0]
        if run_name not in run_names:
            run_names.append(run_name)

    # Second step: Combine results with the same run name
    combined_performance = []
    for run_name in run_names:
        # Get all results with the same run name
        run_results = [result for result in results if result["run_name"].startswith(run_name)]
        # Combine the results
        combined_result = {
            "run_name": run_name,
            "rmse": torch.stack([result["rmse"] for result in run_results]),
            "uce": torch.stack([result["uce"] for result in run_results]),
            "nll": torch.stack([result["nll"] for result in run_results]),
            "train_time": torch.stack([torch.tensor(result["train_time"]) for result in run_results]),
        }
        combined_performance.append(combined_result)

    print(f"Found {len(combined_performance)} unique run_names!")

    # append means of rmse and uce to combined performance and sort by rmse
    for i in range(len(combined_performance)):
        combined_performance[i]["rmse_mean"] = torch.mean(combined_performance[i]["rmse"])
        combined_performance[i]["rmse_std"] = torch.std(combined_performance[i]["rmse"])
        combined_performance[i]["uce_mean"] = torch.mean(combined_performance[i]["uce"])
        combined_performance[i]["uce_std"] = torch.std(combined_performance[i]["uce"])
        combined_performance[i]["nll_mean"] = torch.mean(combined_performance[i]["nll"])
        combined_performance[i]["nll_std"] = torch.std(combined_performance[i]["nll"])
        combined_performance[i]["train_time_mean"] = torch.mean(combined_performance[i]["train_time"])
        combined_performance[i]["train_time_std"] = torch.std(combined_performance[i]["train_time"])
    # Sort by rmse_mean
    combined_performance = sorted(combined_performance, key=lambda x: x["uce_mean"])

    #Save the combined performance
    with open(path + "combined_performance.pkl", 'wb') as f:
        pickle.dump(combined_performance, f)

    # Print the first five candidates
    print("All candidates based on RMSE:")
    for candidate in combined_performance:
        print(f"Run Name: {candidate['run_name']}, RMSE: {candidate['rmse_mean'].item():.4f} ± {candidate['rmse'].std(dim=0).item():.4f}, UCE: {candidate['uce_mean'].item():.4f} ± {candidate['uce'].std(dim=0).item():.4f}, NLL: {candidate['nll_mean'].item():.4f} ± {candidate['nll'].std(dim=0).item():.4f}, Train Time: {candidate['train_time_mean'].item():.4f} ± {candidate['train_time'].std(dim=0).item():.4f}")
    #Print the results to a txt file
    with open(path + "results.txt", 'w') as f:
        for candidate in combined_performance:
            f.write(f"Run Name: {candidate['run_name']}, RMSE: {candidate['rmse_mean'].item():.4f} ± {candidate['rmse'].std(dim=0).item():.4f}, UCE: {candidate['uce_mean'].item():.4f} ± {candidate['uce'].std(dim=0).item():.4f}, NLL: {candidate['nll_mean'].item():.4f} ± {candidate['nll'].std(dim=0).item():.4f}, Train Time: {candidate['train_time_mean'].item():.4f} ± {candidate['train_time'].std(dim=0).item():.4f}\n")
    
    #******Plotting******
    # Structure: "Results"-List has all the raw results, "Combined Performances" has all metrics and the mean metrics as well. Can be accessed through run name

    # Results to be visualized, needs to be specified with _idx-
    #plot_list = ["Progressive_NP25000_progressive_lcd_sequential_L_LEARNED__A0.0000_P1.0000_idx1","KBNN_A0.000100_P1.0000_idx1","MCMC_NS500_L9.0000_P1.0_idx1"]
    plot_list = ["Progressive_NP25000_progressive_lcd_sequential_L9.0000_A0.0000_P1.0000_idx1","KBNN_A0.000100_P1.0000_idx1","MCMC_NS500_L9.0000_P1.0000_idx1"]
    # Uncomment for 800 PT result
    #label_list = ["Progressive, RMSE: 3.39$\pm$0.3, UCE: 2.01$\pm$0.8","KBNN, RMSE: 3.92$\pm$0.4, UCE: 9.49$\pm$1.5","MCMC, RMSE: 3.17$\pm$0.2, UCE: 1.45$\pm$0.7"]
    # Uncomment for 100 PT result
    label_list = ["Progressive","KBNN","MCMC"]
    # Initialite the figure
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })

    plt.rcParams["lines.linewidth"] = 2

    # Rescale x_ext back to original range for plotting
    x_ext = x_scaler.inverse_transform(x_ext.unsqueeze(-1)).squeeze()
    x = x_scaler.inverse_transform(x.unsqueeze(-1)).squeeze()

    factor = 1.6

    fig, ax = plt.subplots(figsize=(factor*3.45, factor*2.2))
    ax.set_title("Regression Results for $x^3$ Benchmark")
    ax.set_xlabel("$x$")
    ax.set_ylabel("Prediction / Ground Truth")

    # PLot Ground truth and true variance
    ax.plot(x_ext, y_gt_ext, label="Ground Truth", color="black", alpha=0.8)
    ax.fill_between(x_ext, y_gt_ext.flatten() - 2*torch.sqrt(torch.tensor(data_noise)),y_gt_ext.flatten() + 2*torch.sqrt(torch.tensor(data_noise)), alpha=0.3, color="black")
    #ax.scatter(x, y, label="Training Data", color="red", alpha=0.2,marker='x')

    for i,plot in enumerate(plot_list):
        # Load the result
        with open(path + plot + '.pkl', 'rb') as f:
            result = pickle.load(f)
        # Plot the results
        ax.plot(x_ext, result["ext_mean_pred"], label=label_list[i], alpha=0.8)
        ax.fill_between(x_ext, result["ext_mean_pred"].flatten() - 2*torch.sqrt(result["ext_var_pred"].flatten()),result["ext_mean_pred"].flatten() + 2*torch.sqrt(result["ext_var_pred"].flatten()), alpha=0.3)

    ax.set_ylim(-75,75)
    ax.set_xlim(-5,5)

    plt.legend(loc='upper left')
    plt.tight_layout()

    plt.savefig(path + "regression_results_cubic.pdf",bbox_inches='tight')
    plt.savefig(path + "regression_results_cubic.svg",bbox_inches='tight')
