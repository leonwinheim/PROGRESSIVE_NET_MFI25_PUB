##################################################################################### 
# Script to demonstrate the effect of incresing number of samples used in progressive
# and the differene in Origin (LCD vs. Random)
# Author: Leon Winheim, ISAS at KIT
######################################################################################
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

# Path to save the data
path="temp/sample_performance_400/"

if not os.path.exists(path):
    os.makedirs(path)

#******Data Generation******
# Variance of the noisy data
data_noise = 9

# Range definition for training
num_points = 400
border = 4

# Noiseless model
def true_model(x):
    return x**3
# Noisy model
def noisy_model(x,noise):
    return true_model(x) + torch.sqrt(torch.tensor(noise)) * torch.randn(x.size())

# Define trainning range
x = torch.linspace(-border,border,num_points)
x_ext = torch.linspace(-1.5*border,1.5*border,num_points)

# Generate Data
y_gt = true_model(x)                # Ground Truth
y_gt_ext = true_model(x_ext)        # Ground Truth extended
y = noisy_model(x,data_noise)       # Noisy, for Training /Testing

# Normalize input data to be between -1 and 1
x_scaler  = preprocessing.MinMaxScaler(feature_range=(-1,1))
x = torch.tensor(x_scaler.fit_transform(x.unsqueeze(-1)),dtype=torch.float32).squeeze()
x_ext = torch.tensor(x_scaler.transform(x_ext.unsqueeze(-1)), dtype=torch.float32).squeeze()

#******Run environment******
number_runs = 10

perm_list = []
# Set the shuffled indices for each run beforehand so they stay the same
for i in range(number_runs):
    perm = torch.randperm(num_points)
    perm_list.append(perm)

for i in range(number_runs):
    #******Data Shuffling******
    indices = perm_list[i]
    x = x[indices]
    y = y[indices]

    #******Train/Test Split******
    split_ratio = 0.8   
    x_train = x[:int(split_ratio*num_points)]
    y_train = y[:int(split_ratio*num_points)]
    x_test = x[int(split_ratio*num_points):]
    y_test = y[int(split_ratio*num_points):]

    #******Architecture Parameters******
    layers = [1,20,1]
    act_func = ["relu","linear"]

    #******Assemble Run Parameters******
    progressive_pars = []

    algo  = ["progressive_lcd","progressive"]
    training_type = ["sequential"]
    num_particles = [1000,2000,3000,4000,5000,7500,10000,20000]

    data_noise_low = 9.0
    data_noise_high = 9.0
    data_noise_step = 1.0

    artificial_noise_low = 0.0
    artificial_noise_high = 0.0
    artificial_noise_step = 0.001

    prior_noise_low = 1.0
    prior_noise_high = 1.0
    prior_noise_step = 1

    # Create the parameter grid
    for algo_type in algo:
        for num_part in num_particles:
            for train_type in training_type:
                for data_noise in np.arange(data_noise_low, data_noise_high + data_noise_step, data_noise_step):
                    for artificial_noise in np.arange(artificial_noise_low, artificial_noise_high + artificial_noise_step, artificial_noise_step):
                        for prior_noise in np.arange(prior_noise_low, prior_noise_high + prior_noise_step, prior_noise_step):
                            progressive_pars.append((x_train,y_train,x_test,y_test,layers,act_func,
                                                algo_type,train_type,num_part,data_noise,artificial_noise,prior_noise,i))

    print("***********************************************")
    print("Run number: ", i)
    print(f"Parameters for progressive:{len(progressive_pars)}")
    print("***********************************************")

    #******Perform the algorithms******
    for j, parset in enumerate(progressive_pars, start=1):
        # Check wether run is already made
        run_name = f"progressive_NP{parset[8]}_{parset[6]}_{parset[7]}_L{parset[9]:.4f}_A{parset[10]:.4f}_P{parset[11]:.4f}_idx{i}"
        if os.path.exists(path + run_name + '.pkl'):
            print(f"Run {run_name} already exists. Skipping...")
            continue
        # Run
        print(f"Running progressive parameter set {j}/{len(progressive_pars)}")
        result = rf.run_PROGRESSIVE(*parset)
        if not result is None:
            with open(path+result["run_name"]+'.pkl', 'wb') as f:
                pickle.dump(result, f)
        else:
            print(f"progressive failed for {parset[0]}!")

        print("Finished all runs!")

#******Evaluation****** 
#Generate a list of pkl files in the directory
files = os.listdir(path)
files = [f for f in files if (f.endswith(".pkl"))]
files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(path, x)))

if files == []:
    print("No files found")
    exit()

#Evaluate and save
results = []
for file in files:
    results.append(hf.evaluate(path,file))

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
# Sort by rmse_mean
combined_performance = sorted(combined_performance, key=lambda x: x["uce_mean"])

# Print the first five candidates
print("All candidates based on RMSE:")
for candidate in combined_performance:
    print(f"Run Name: {candidate['run_name']}, RMSE Mean: {candidate['rmse_mean']:.4f}, UCE Mean: {candidate['uce_mean']:.4f}, NLL Mean: {candidate['nll_mean']:.4f}")

# Structure: "Results"-List has all the raw results, "Combined Performances" has all metrics and the mean metrics as well. Can be accessed through run name

#******Plotting******

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

plt.rcParams["lines.linewidth"] = 2  # Set default line width for all plots

visibility_offset = 100

# Plot Performance vs Sample count
factor = 1.2
plt.figure(figsize=(factor*4, factor*4))

# Subplot 1: UCE vs Sample Count
plt.subplot(2, 1, 1)
sample_counts = []
uce_means = []
uce_stds = []

# Do it for the LCD Samples
for candidate in combined_performance:
    if "progressive_lcd" in candidate["run_name"]:
        sample_count = int(candidate["run_name"].split("NP")[1].split("_")[0])  # Extract sample count
        sample_counts.append(sample_count)
        uce_means.append(candidate["uce_mean"].item())
        uce_stds.append(candidate["uce_std"].item())

# Sort by sample count for a cleaner plot
sorted_indices = np.argsort(sample_counts)
sample_counts = np.array(sample_counts)[sorted_indices]
uce_means = np.array(uce_means)[sorted_indices]
uce_stds = np.array(uce_stds)[sorted_indices]

color = plt.get_cmap("tab10")(0)  # tab blue
plt.plot(sample_counts-visibility_offset, uce_means, marker='o', label="LCD samples", alpha=0.7, color=color)
plt.errorbar(sample_counts-visibility_offset, uce_means, yerr=uce_stds, fmt='o', alpha=0.5, capsize=5,color=color)

sample_counts = []
uce_means = []
uce_stds = []

# Do it for the Random Samples
for candidate in combined_performance:
    if "progressive_sequential" in candidate["run_name"]:
        sample_count = int(candidate["run_name"].split("NP")[1].split("_")[0])  # Extract sample count
        sample_counts.append(sample_count)
        uce_means.append(candidate["uce_mean"].item())
        uce_stds.append(candidate["uce_std"].item())

# Sort by sample count for a cleaner plot
sorted_indices = np.argsort(sample_counts)
sample_counts = np.array(sample_counts)[sorted_indices]
uce_means = np.array(uce_means)[sorted_indices]
uce_stds = np.array(uce_stds)[sorted_indices]

color = plt.get_cmap("tab10")(1)  # tab orange
plt.plot(sample_counts+visibility_offset, uce_means, marker='o', label="Random samples", alpha=0.7,color=color)
plt.errorbar(sample_counts+visibility_offset, uce_means, yerr=uce_stds, fmt='o', alpha=0.5, capsize=5,color=color)

plt.title("UCE vs. Sample Count for progressive net")
plt.xlabel("Sample Count")
plt.ylabel("UCE")
plt.grid(True)
plt.ylim(0,)
plt.legend(loc="upper right", fontsize="small")

# Subplot 2: RMSE vs Sample Count
plt.subplot(2, 1, 2)
sample_counts = []
rmse_means = []
rmse_stds = []

# Do it for the LCD Samples
for candidate in combined_performance:
    if "progressive_lcd" in candidate["run_name"]:
        sample_count = int(candidate["run_name"].split("NP")[1].split("_")[0])  # Extract sample count
        sample_counts.append(sample_count)
        rmse_means.append(candidate["rmse_mean"].item())
        rmse_stds.append(candidate["rmse_std"].item())

# Sort by sample count for a cleaner plot
sorted_indices = np.argsort(sample_counts)
sample_counts = np.array(sample_counts)[sorted_indices]
rmse_means = np.array(rmse_means)[sorted_indices]
rmse_stds = np.array(rmse_stds)[sorted_indices]

color = plt.get_cmap("tab10")(0)
plt.plot(sample_counts-visibility_offset, rmse_means, marker='o', label="LCD samples", alpha=0.7,color=color)
plt.errorbar(sample_counts-visibility_offset, rmse_means, yerr=rmse_stds, fmt='o', alpha=0.5, capsize=5,color=color)

sample_counts = []
rmse_means = []
rmse_stds = []

# Do it for the Random Samples
for candidate in combined_performance:
    if "progressive_sequential" in candidate["run_name"]:
        sample_count = int(candidate["run_name"].split("NP")[1].split("_")[0])  # Extract sample count
        sample_counts.append(sample_count)
        rmse_means.append(candidate["rmse_mean"].item())
        rmse_stds.append(candidate["rmse_std"].item())

# Sort by sample count for a cleaner plot
sorted_indices = np.argsort(sample_counts)
sample_counts = np.array(sample_counts)[sorted_indices]
rmse_means = np.array(rmse_means)[sorted_indices]
rmse_stds = np.array(rmse_stds)[sorted_indices]

color = plt.get_cmap("tab10")(1)
plt.plot(sample_counts+visibility_offset, rmse_means, marker='o', label="Random samples", alpha=0.7,color=color)
plt.errorbar(sample_counts+visibility_offset, rmse_means, yerr=rmse_stds, fmt='o', alpha=0.5, capsize=5,color=color)

plt.title("RMSE vs. Sample Count for progressive net")
plt.xlabel("Sample Count")
plt.ylabel("RMSE ")
plt.grid(True)
plt.legend(loc="upper right", fontsize="small")

plt.tight_layout()
plt.savefig(path + "sample_performance.pdf", bbox_inches='tight')
plt.savefig(path + "sample_performance.svg", bbox_inches='tight')

