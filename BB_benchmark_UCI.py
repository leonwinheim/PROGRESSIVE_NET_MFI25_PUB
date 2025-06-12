#############################################################################
# Evaluation of Regression performance on UCI Datasets
# comparing PROGRESSIVE, KBNN and MCMC
# Author: Leon Winheim
#############################################################################
import torch
import numpy as np
import os
import ZZ_run_funcs as rf
import ZZ_help_funcs as hf
import pickle
import matplotlib.pyplot as plt
import pandas as pd

#******Flags******
save_datasets = False

run_concrete = False    #Check
run_energy = False      #Check
run_naval_propulsion = True
run_yacht = False   #Check
run_kin8nm = False #Check
run_power = False   #Semigut

if save_datasets:
    #******Generate CSV-Files with UCI Datasets******
    datasets = ["dataset_concrete", "dataset_energy",
                    "dataset_naval_propulsion", "dataset_yacht",
                    "dataset_kin8nm","dataset_power"]

    for dataset in datasets:
        hf.save_uci_data(dataset)

#******Control Variables******
number_runs = 10

base_path = "temp/UCI/"
if not os.path.exists(base_path):
    os.makedirs(base_path)

#******Run Simulations******
if run_concrete:
    print()
    print("Running Concrete Dataset...")
    print()
    #******Control Variables******
    torch.manual_seed(41)
    np.random.seed(41)

    # Path to save the data
    path_save = base_path + "concrete/"
    # Path to get the data from
    path_load = "datasets/"

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    #******Data Loading******	
    name = "dataset_concrete_data"
    with open(os.path.join(path_load, f"{name}.pkl"), "rb") as f:
        data_dict = pickle.load(f)
    X = data_dict["X"]
    Y = data_dict["Y"]
    x_scaler = data_dict["x_scaler"]
    y_scaler = data_dict["y_scaler"]

    # Convert to torch tensors
    x = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(Y, dtype=torch.float32)

    # Ensure y is 2D
    if y.ndimension() == 1:
        y = y.unsqueeze(1)

    print("Number of Features: ", x.shape[1])
    print("Number of Samples: ", x.shape[0])
    print("Number of Outputs: ", y.shape[0])

    #******Run environment******
    perm_list = []
    # Set the shuffled indices for each run beforehand so they stay the same
    for i in range(number_runs):
        perm = torch.randperm(x.size(0))
        perm_list.append(perm)

    for i in range(number_runs):
        #******Data Shuffling****** 
        indices = perm_list[i]
        x = x[indices]
        y = y[indices]

        #******Data Splitting******
        split_rate = 0.8
        split = int(split_rate * x.size(0))
        x_train = x[:split]
        y_train = y[:split]
        x_test = x[split:]
        y_test = y[split:]

        #******Architecture Parameters******
        # Define the parameters for my one hidden layer network (Relu Activated)
        hidden_units = 50
        layers = [x.shape[1],hidden_units,1]
        act_func = ["relu","linear"]

        #******Assemble Run Parameters******
        # PROGRESSIVE
        progressive_pars = []

        algo  = ["progressive_lcd"]
        training_type = ["sequential"]
        num_particles = [25000]

        data_noise = None

        artificial_noise_low = 0.00
        artificial_noise_high = 0.00
        artificial_noise_step = 0.01

        prior_noise_low = 1.0
        prior_noise_high = 1.0
        prior_noise_step = 0.5

        # Create the parameter grid
        for algo_type in algo:
            for num_part in num_particles:
                for train_type in training_type:
                    for artificial_noise in np.arange(artificial_noise_low, artificial_noise_high + artificial_noise_step, artificial_noise_step):
                        for prior_noise in np.arange(prior_noise_low, prior_noise_high + prior_noise_step, prior_noise_step):
                            progressive_pars.append((x_train,y_train,x_test,y_test,layers,act_func,
                                                algo_type,train_type,num_part,None,artificial_noise,prior_noise,i))
        # KBNN
        kbnn_pars = []

        artificial_noise_low = 0.066
        artificial_noise_high = 0.066
        artificial_noise_step = 10

        prior_noise_low = 1.0
        prior_noise_high = 1.0
        prior_noise_step = 10

        #Create parameter grid
        for artificial_noise in np.arange(artificial_noise_low, artificial_noise_high + artificial_noise_step, artificial_noise_step):
            for prior_noise in np.arange(prior_noise_low, prior_noise_high + prior_noise_step, prior_noise_step):
                kbnn_pars.append((x_train,y_train,x_test,y_test,layers,act_func,
                                artificial_noise,prior_noise,i))
        # MCMC
        mcmc_pars = []
        
        num_samples = 100

        data_noise = None

        prior_noise_low = 1.0
        prior_noise_high = 1.0
        prior_noise_step = 10

        # Create the parameter grid
        for prior_noise in np.arange(prior_noise_low, prior_noise_high + prior_noise_step*0.9, prior_noise_step):
            mcmc_pars.append((x_train,y_train,x_test,y_test,layers[1],prior_noise,
                                num_samples,data_noise,i))

        for j, parset in enumerate(progressive_pars, start=1):
            # Check wether run is already made
            if parset[9] is None:
                run_name = f"Progressive_NP{parset[8]}_{parset[6]}_{parset[7]}_L{'_LEARNED_'}_A{parset[10]:.4f}_P{parset[11]:.4f}_idx{i}"
            else:
                run_name = f"Progressive_NP{parset[8]}_{parset[6]}_{parset[7]}_L{parset[9]:.4f}_A{parset[10]:.4f}_P{parset[11]:.4f}_idx{i}"
            if os.path.exists(path_save + run_name + '.pkl'):
                print(f"Run {run_name} already exists. Skipping...")
                continue
            # Run
            print(f"Running Progressive parameter set {j}/{len(progressive_pars)}")
            result = rf.run_PROGRESSIVE(*parset,meas_guess=1.0)
            if not result is None:
                with open(path_save+result["run_name"]+'.pkl', 'wb') as f:
                    pickle.dump(result, f)
            else:
                print(f"Progressive failed for {parset[0]}!")

        for j, parset in enumerate(kbnn_pars, start=1):
            # Check wether run is already made
            run_name = f"KBNN_A{parset[6]:.6f}_P{parset[7]:.4f}_idx{i}"
            if os.path.exists(path_save + run_name + '.pkl'):
                print(f"Run {run_name} already exists. Skipping...")
                continue
            # Run
            print(f"Running KBNN parameter set {j}/{len(kbnn_pars)}")
            result = rf.run_KBNN(*parset)
            if not result is None:
                with open(path_save+result["run_name"]+'.pkl', 'wb') as f:
                    pickle.dump(result, f)
            else:
                print(f"KBNN failed for {parset[0]}!")

        for j, parset in enumerate(mcmc_pars, start=1):
            # Check wether run is already made
            if parset[7] is None:
                run_name = f"MCMC_NS{parset[6]}_L{'_LEARNED_'}_P{parset[5]:.4f}_idx{i}"
            else:
                run_name = f"MCMC_NS{parset[6]}_L{parset[7]:.4f}_P{parset[5]:.4f}_idx{i}"
            if os.path.exists(path_save + run_name + '.pkl') :
                print(f"Run {run_name} already exists. Skipping...")
                continue
            # Run
            print(f"Running MCMC parameter set {j}/{len(mcmc_pars)} "+run_name)
            result = rf.run_MCMC(*parset)
            if not result is None:
                with open(path_save+result["run_name"]+'.pkl', 'wb') as f:
                    pickle.dump(result, f)
            else:
                print(f"MCMC failed for {parset[0]}!")

    #******Evaluation******

    #Generate a list of pkl files in the directory
    files = os.listdir(path_save)
    files = [f for f in files if (f.endswith(".pkl"))]
    files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(path_save, x)))

    if files == []:
        print("No files found")
        exit()

    #Evaluate and save
    results = []
    for file in files:
        results.append(hf.evaluate(path_save,file))

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
        combined_performance[i]["uce_mean"] = torch.mean(combined_performance[i]["uce"])
        combined_performance[i]["nll_mean"] = torch.mean(combined_performance[i]["nll"])
        combined_performance[i]["train_time_mean"] = torch.mean(combined_performance[i]["train_time"])
    # Sort by rmse_mean
    combined_performance = sorted(combined_performance, key=lambda x: x["rmse_mean"])

    # Print Results
    print("All candidates based on RMSE:")
    for candidate in combined_performance:
        print(f"Run Name: {candidate['run_name']}, RMSE: {candidate['rmse_mean'].item():.4f} ± {candidate['rmse'].std(dim=0).item():.4f}, UCE: {candidate['uce_mean'].item():.4f} ± {candidate['uce'].std(dim=0).item():.4f}, NLL: {candidate['nll_mean'].item():.4f} ± {candidate['nll'].std(dim=0).item():.4f}, Train Time: {candidate['train_time_mean'].item():.4f} ± {candidate['train_time'].std(dim=0).item():.4f}")

    # Save the results in a txt file
    with open(path_save + "results.txt", "w") as f:
        for candidate in combined_performance:
            f.write(f"Run Name: {candidate['run_name']}, RMSE: {candidate['rmse_mean'].item():.4f} ± {candidate['rmse'].std(dim=0).item():.4f}, UCE: {candidate['uce_mean'].item():.4f} ± {candidate['uce'].std(dim=0).item():.4f}, NLL: {candidate['nll_mean'].item():.4f} ± {candidate['nll'].std(dim=0).item():.4f}, Train Time: {candidate['train_time_mean'].item():.4f} ± {candidate['train_time'].std(dim=0).item():.4f}\n")

if run_energy:
    print()
    print("Running Energy Dataset...")
    print()
    #******Control Variables******
    torch.manual_seed(41)
    np.random.seed(41)

    # Path to save the data
    path_save = base_path + "energy/"
    # Path to get the data from
    path_load = "datasets/"

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    #******Data Loading******	
    name = "dataset_energy_data"
    with open(os.path.join(path_load, f"{name}.pkl"), "rb") as f:
        data_dict = pickle.load(f)
    X = data_dict["X"]
    Y = data_dict["Y"]
    x_scaler = data_dict["x_scaler"]
    y_scaler = data_dict["y_scaler"]

    # Convert to torch tensors
    x = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(Y, dtype=torch.float32)

    # Ensure y is 2D
    if y.ndimension() == 1:
        y = y.unsqueeze(1)

    print("Number of Features: ", x.shape[1])
    print("Number of Samples: ", x.shape[0])
    print("Number of Outputs: ", y.shape[0])

    #******Run environment******
    perm_list = []
    # Set the shuffled indices for each run beforehand so they stay the same
    for i in range(number_runs):
        perm = torch.randperm(x.size(0))
        perm_list.append(perm)

    for i in range(number_runs):
        #******Data Shuffling******
        indices = perm_list[i]
        x = x[indices]
        y = y[indices]

        #******Data Splitting******
        split_rate = 0.8
        split = int(split_rate * x.size(0))
        x_train = x[:split]
        y_train = y[:split]
        x_test = x[split:]
        y_test = y[split:]

        #******Architecture Parameters******
        # Define the parameters for my one hidden layer network (Relu Activated)
        hidden_units = 50
        layers = [x.shape[1],hidden_units,1]
        act_func = ["relu","linear"]

        #******Assemble Run Parameters******
        # PROGRESSIVE
        progressive_pars = []

        algo  = ["progressive_lcd"]
        training_type = ["sequential"]
        num_particles = [25000]

        data_noise = None

        artificial_noise_low = 0.00
        artificial_noise_high = 0.00
        artificial_noise_step = 0.05

        prior_noise_low = 1.0
        prior_noise_high = 1.0
        prior_noise_step = 0.5

        # Create the parameter grid
        for algo_type in algo:
            for num_part in num_particles:
                for train_type in training_type:
                    for artificial_noise in np.arange(artificial_noise_low, artificial_noise_high + artificial_noise_step, artificial_noise_step):
                        for prior_noise in np.arange(prior_noise_low, prior_noise_high + prior_noise_step, prior_noise_step):
                            progressive_pars.append((x_train,y_train,x_test,y_test,layers,act_func,
                                                algo_type,train_type,num_part,None,artificial_noise,prior_noise,i))

        # KBNN
        kbnn_pars = []

        artificial_noise_low = 0.451
        artificial_noise_high = 0.451
        artificial_noise_step = 0.01

        prior_noise_low = 1.0
        prior_noise_high = 1.0
        prior_noise_step = 0.1

        #Create parameter grid
        for artificial_noise in np.arange(artificial_noise_low, artificial_noise_high + artificial_noise_step, artificial_noise_step):
            for prior_noise in np.arange(prior_noise_low, prior_noise_high + prior_noise_step, prior_noise_step):
                kbnn_pars.append((x_train,y_train,x_test,y_test,layers,act_func,
                                artificial_noise,prior_noise,i))

        # MCMC
        mcmc_pars = []
        
        num_samples = 100
        data_noise = None

        prior_noise_low = 1.0
        prior_noise_high = 1.0
        prior_noise_step = 10

        # Create the parameter grid
        for prior_noise in np.arange(prior_noise_low, prior_noise_high + prior_noise_step*0.9, prior_noise_step):
            mcmc_pars.append((x_train,y_train,x_test,y_test,layers[1],prior_noise,
                                num_samples,data_noise,i))

        for j, parset in enumerate(progressive_pars, start=1):
            # Check wether run is already made
            if parset[9] is None:
                run_name = f"Progressive_NP{parset[8]}_{parset[6]}_{parset[7]}_L{'_LEARNED_'}_A{parset[10]:.4f}_P{parset[11]:.4f}_idx{i}"
            else:
                run_name = f"Progressive_NP{parset[8]}_{parset[6]}_{parset[7]}_L{parset[9]:.4f}_A{parset[10]:.4f}_P{parset[11]:.4f}_idx{i}"
            if os.path.exists(path_save + run_name + '.pkl'):
                print(f"Run {run_name} already exists. Skipping...")
                continue
            # Run
            print(f"Running Progressive parameter set {j}/{len(progressive_pars)}")
            result = rf.run_PROGRESSIVE(*parset,meas_guess=1.0)
            if not result is None:
                with open(path_save+result["run_name"]+'.pkl', 'wb') as f:
                    pickle.dump(result, f)
            else:
                print(f"Progressive failed for {parset[0]}!")

        for j, parset in enumerate(kbnn_pars, start=1):
            # Check wether run is already made
            run_name = f"KBNN_A{parset[6]:.6f}_P{parset[7]:.4f}_idx{i}"
            if os.path.exists(path_save + run_name + '.pkl'):
                print(f"Run {run_name} already exists. Skipping...")
                continue
            # Run
            print(f"Running KBNN parameter set {j}/{len(kbnn_pars)}")
            result = rf.run_KBNN(*parset)
            if not result is None:
                with open(path_save+result["run_name"]+'.pkl', 'wb') as f:
                    pickle.dump(result, f)
            else:
                print(f"KBNN failed for {parset[0]}!")

        for j, parset in enumerate(mcmc_pars, start=1):
            # Check wether run is already made
            if parset[7] is None:
                run_name = f"MCMC_NS{parset[6]}_L{'_LEARNED_'}_P{parset[5]:.4f}_idx{i}"
            else:
                run_name = f"MCMC_NS{parset[6]}_L{parset[7]:.4f}_P{parset[5]:.4f}_idx{i}"
            if os.path.exists(path_save + run_name + '.pkl') :
                print(f"Run {run_name} already exists. Skipping...")
                continue
            # Run
            print(f"Running MCMC parameter set {j}/{len(mcmc_pars)} "+run_name)
            result = rf.run_MCMC(*parset)
            if not result is None:
                with open(path_save+result["run_name"]+'.pkl', 'wb') as f:
                    pickle.dump(result, f)
            else:
                print(f"MCMC failed for {parset[0]}!")

    #******Evaluation******
    #Generate a list of pkl files in the directory
    files = os.listdir(path_save)
    files = [f for f in files if (f.endswith(".pkl"))]
    files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(path_save, x)))

    if files == []:
        print("No files found")
        exit()

    #Evaluate and save
    results = []
    for file in files:
        results.append(hf.evaluate(path_save,file))

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
        combined_performance[i]["uce_mean"] = torch.mean(combined_performance[i]["uce"])
        combined_performance[i]["nll_mean"] = torch.mean(combined_performance[i]["nll"])
        combined_performance[i]["train_time_mean"] = torch.mean(combined_performance[i]["train_time"])
    # Sort by rmse_mean
    combined_performance = sorted(combined_performance, key=lambda x: x["rmse_mean"])

    # Print the first five candidates
    print("All candidates based on RMSE:")
    for candidate in combined_performance:
        print(f"Run Name: {candidate['run_name']}, RMSE: {candidate['rmse_mean'].item():.4f} ± {candidate['rmse'].std(dim=0).item():.4f}, UCE: {candidate['uce_mean'].item():.4f} ± {candidate['uce'].std(dim=0).item():.4f}, NLL: {candidate['nll_mean'].item():.4f} ± {candidate['nll'].std(dim=0).item():.4f}, Train Time: {candidate['train_time_mean'].item():.4f} ± {candidate['train_time'].std(dim=0).item():.4f}")
    # Save the results in a txt file
    with open(path_save + "results.txt", "w") as f:
        for candidate in combined_performance:
            f.write(f"Run Name: {candidate['run_name']}, RMSE: {candidate['rmse_mean'].item():.4f} ± {candidate['rmse'].std(dim=0).item():.4f}, UCE: {candidate['uce_mean'].item():.4f} ± {candidate['uce'].std(dim=0).item():.4f}, NLL: {candidate['nll_mean'].item():.4f} ± {candidate['nll'].std(dim=0).item():.4f}, Train Time: {candidate['train_time_mean'].item():.4f} ± {candidate['train_time'].std(dim=0).item():.4f}\n")

if run_naval_propulsion:
    print()
    print("Running Naval Propulsion Dataset...")
    print()
    #******Control Variables******
    torch.manual_seed(40)
    np.random.seed(40)

    # Path to save the data
    path_save = base_path + "naval_propulsion/"
    # Path to get the data from
    path_load = "datasets/"

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    #******Data Loading******	
    name = "dataset_naval_propulsion_data"
    with open(os.path.join(path_load, f"{name}.pkl"), "rb") as f:
        data_dict = pickle.load(f)
    X = data_dict["X"]
    Y = data_dict["Y"]
    x_scaler = data_dict["x_scaler"]
    y_scaler = data_dict["y_scaler"]

    # Convert to torch tensors
    x = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(Y, dtype=torch.float32)

    # Ensure y is 2D
    if y.ndimension() == 1:
        y = y.unsqueeze(1)

    print("Number of Features: ", x.shape[1])
    print("Number of Samples: ", x.shape[0])
    print("Number of Outputs: ", y.shape[0])

    #******Run environment******
    perm_list = []
    # Set the shuffled indices for each run beforehand so they stay the same
    for i in range(number_runs):
        perm = torch.randperm(x.size(0))
        perm_list.append(perm)

    for i in range(number_runs):
        #******Data Shuffling******
        indices = perm_list[i]
        x = x[indices]
        y = y[indices]
        
        #******Data Splitting******
        split_rate = 0.8
        split = int(split_rate * x.size(0))
        x_train = x[:split]
        y_train = y[:split]
        x_test = x[split:]
        y_test = y[split:]

        #******Architecture Parameters******
        # Define the parameters for my one hidden layer network (Relu Activated)
        hidden_units = 50
        layers = [x.shape[1],hidden_units,1]
        act_func = ["relu","linear"]

        #******Assemble Run Parameters******
        # PROGRESSIVE
        progressive_pars = []

        algo  = ["progressive_lcd"]
        training_type = ["sequential"]
        num_particles = [25000]

        #data_noise = 0.00025
        data_noise = 0.1

        artificial_noise_low = 0.0
        artificial_noise_high = 0.0
        artificial_noise_step = 0.05

        prior_noise_low = 1.0
        prior_noise_high = 1.0
        prior_noise_step = 0.5

        # Create the parameter grid
        for algo_type in algo:
            for num_part in num_particles:
                for train_type in training_type:
                    for artificial_noise in np.arange(artificial_noise_low, artificial_noise_high + artificial_noise_step, artificial_noise_step):
                        for prior_noise in np.arange(prior_noise_low, prior_noise_high + prior_noise_step, prior_noise_step):
                            progressive_pars.append((x_train,y_train,x_test,y_test,layers,act_func,
                                                algo_type,train_type,num_part,data_noise,artificial_noise,prior_noise,i))
        
        # KBNN
        kbnn_pars = []

        artificial_noise_low = 0.026
        artificial_noise_high = 0.026
        artificial_noise_step = 0.005

        prior_noise_low = 1.0
        prior_noise_high = 1.0
        prior_noise_step = 0.1

        #Create parameter grid
        for artificial_noise in np.arange(artificial_noise_low, artificial_noise_high + artificial_noise_step*0.9, artificial_noise_step):
            for prior_noise in np.arange(prior_noise_low, prior_noise_high + prior_noise_step*0.9, prior_noise_step):
                kbnn_pars.append((x_train,y_train,x_test,y_test,layers,act_func,
                                artificial_noise,prior_noise,i))

        # MCMC
        mcmc_pars = []
        
        num_samples = 100
        data_noise = None

        prior_noise_low = 1.0
        prior_noise_high = 1.0
        prior_noise_step = 10

        # Create the parameter grid
        for prior_noise in np.arange(prior_noise_low, prior_noise_high + prior_noise_step*0.9, prior_noise_step):
            mcmc_pars.append((x_train,y_train,x_test,y_test,layers[1],prior_noise,
                                num_samples,data_noise,i))

        for j, parset in enumerate(progressive_pars, start=1):
            # Check wether run is already made
            if parset[9] is None:
                run_name = f"Progressive_NP{parset[8]}_{parset[6]}_{parset[7]}_L{'_LEARNED_'}_A{parset[10]:.4f}_P{parset[11]:.4f}_idx{i}"
            else:
                run_name = f"Progressive_NP{parset[8]}_{parset[6]}_{parset[7]}_L{parset[9]:.4f}_A{parset[10]:.4f}_P{parset[11]:.4f}_idx{i}"
            if os.path.exists(path_save + run_name + '.pkl'):
                print(f"Run {run_name} already exists. Skipping...")
                continue
            # Run
            print(f"Running Progressive parameter set {j}/{len(progressive_pars)}")
            result = rf.run_PROGRESSIVE(*parset,meas_guess=None)
            if not result is None:
                with open(path_save+result["run_name"]+'.pkl', 'wb') as f:
                    pickle.dump(result, f)
            else:
                print(f"Progressive failed for {parset[0]}!")

        for j, parset in enumerate(kbnn_pars, start=1):
            # Check wether run is already made
            run_name = f"KBNN_A{parset[6]:.6f}_P{parset[7]:.4f}_idx{i}"
            if os.path.exists(path_save + run_name + '.pkl'):
                print(f"Run {run_name} already exists. Skipping...")
                continue
            # Run
            print(f"Running KBNN parameter set {j}/{len(kbnn_pars)}")
            result = rf.run_KBNN(*parset)
            if not result is None:
                with open(path_save+result["run_name"]+'.pkl', 'wb') as f:
                    pickle.dump(result, f)
            else:
                print(f"KBNN failed for {parset[0]}!")


        for j, parset in enumerate(mcmc_pars, start=1):
            # Check wether run is already made
            if parset[7] is None:
                run_name = f"MCMC_NS{parset[6]}_L{'_LEARNED_'}_P{parset[5]:.4f}_idx{i}"
            else:
                run_name = f"MCMC_NS{parset[6]}_L{parset[7]:.4f}_P{parset[5]:.4f}_idx{i}"
            if os.path.exists(path_save + run_name + '.pkl') :
                print(f"Run {run_name} already exists. Skipping...")
                continue
            # Run
            print(f"Running MCMC parameter set {j}/{len(mcmc_pars)} "+run_name)
            result = rf.run_MCMC(*parset)
            if not result is None:
                with open(path_save+result["run_name"]+'.pkl', 'wb') as f:
                    pickle.dump(result, f)
            else:
                print(f"MCMC failed for {parset[0]}!")

    #******Evaluation******
    #Generate a list of pkl files in the directory
    files = os.listdir(path_save)
    files = [f for f in files if (f.endswith(".pkl"))]
    files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(path_save, x)))

    if files == []:
        print("No files found")
        exit()

    #Evaluate and save
    results = []
    for file in files:
        results.append(hf.evaluate(path_save,file))

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
        combined_performance[i]["uce_mean"] = torch.mean(combined_performance[i]["uce"])
        combined_performance[i]["nll_mean"] = torch.mean(combined_performance[i]["nll"])
        combined_performance[i]["train_time_mean"] = torch.mean(combined_performance[i]["train_time"])
    # Sort by rmse_mean
    combined_performance = sorted(combined_performance, key=lambda x: x["rmse_mean"])

    # Print the first five candidates
    print("All candidates based on RMSE:")
    for candidate in combined_performance:
        print(f"Run Name: {candidate['run_name']}, RMSE: {candidate['rmse_mean'].item():.4f} ± {candidate['rmse'].std(dim=0).item():.4f}, UCE: {candidate['uce_mean'].item():.4f} ± {candidate['uce'].std(dim=0).item():.4f}, NLL: {candidate['nll_mean'].item():.4f} ± {candidate['nll'].std(dim=0).item():.4f}, Train Time: {candidate['train_time_mean'].item():.4f} ± {candidate['train_time'].std(dim=0).item():.4f}")
    # Save the results in a txt file
    with open(path_save + "results.txt", "w") as f:
        for candidate in combined_performance:
            f.write(f"Run Name: {candidate['run_name']}, RMSE: {candidate['rmse_mean'].item():.4f} ± {candidate['rmse'].std(dim=0).item():.4f}, UCE: {candidate['uce_mean'].item():.4f} ± {candidate['uce'].std(dim=0).item():.4f}, NLL: {candidate['nll_mean'].item():.4f} ± {candidate['nll'].std(dim=0).item():.4f}, Train Time: {candidate['train_time_mean'].item():.4f} ± {candidate['train_time'].std(dim=0).item():.4f}\n")

if run_kin8nm:
    print()
    print("Running Kin8nm Dataset...")
    print()
    #******Control Variables******
    torch.manual_seed(41)
    np.random.seed(41)

    # Path to save the data
    path_save = base_path + "kin8nm/"
    # Path to get the data from
    path_load = "datasets/"

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    #******Data Loading******	
    name = "dataset_kin8nm_data"
    with open(os.path.join(path_load, f"{name}.pkl"), "rb") as f:
        data_dict = pickle.load(f)
    X = data_dict["X"]
    Y = data_dict["Y"]
    x_scaler = data_dict["x_scaler"]
    y_scaler = data_dict["y_scaler"]

    # Convert to torch tensors
    x = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(Y, dtype=torch.float32)

    # Ensure y is 2D
    if y.ndimension() == 1:
        y = y.unsqueeze(1)

    print("Number of Features: ", x.shape[1])
    print("Number of Samples: ", x.shape[0])
    print("Number of Outputs: ", y.shape[0])


    #******Run environment******
    perm_list = []
    # Set the shuffled indices for each run beforehand so they stay the same
    for i in range(number_runs):
        perm = torch.randperm(x.size(0))
        perm_list.append(perm)

    for i in range(number_runs):
        #******Data Shuffling******
        indices = perm_list[i]
        x = x[indices]
        y = y[indices]

        #******Data Splitting******
        split_rate = 0.8
        split = int(split_rate * x.size(0))
        x_train = x[:split]
        y_train = y[:split]
        x_test = x[split:]
        y_test = y[split:]

        #******Architecture Parameters******
        # Define the parameters for my one hidden layer network (Relu Activated)
        hidden_units = 50
        layers = [x.shape[1],hidden_units,1]
        act_func = ["relu","linear"]

        #******Assemble Run Parameters******
        # PROGRESSIVE
        progressive_pars = []

        algo  = ["progressive_lcd"]
        training_type = ["sequential"]
        num_particles = [25000]

        data_noise = None

        artificial_noise_low = 0.0
        artificial_noise_high = 0.0
        artificial_noise_step = 0.05

        prior_noise_low = 1.0
        prior_noise_high = 1.0
        prior_noise_step = 0.5

        # Create the parameter grid
        for algo_type in algo:
            for num_part in num_particles:
                for train_type in training_type:
                    for artificial_noise in np.arange(artificial_noise_low, artificial_noise_high + artificial_noise_step, artificial_noise_step):
                        for prior_noise in np.arange(prior_noise_low, prior_noise_high + prior_noise_step, prior_noise_step):
                            progressive_pars.append((x_train,y_train,x_test,y_test,layers,act_func,
                                                algo_type,train_type,num_part,data_noise,artificial_noise,prior_noise,i))
        # KBNN
        kbnn_pars = []

        artificial_noise_low = 0.026
        artificial_noise_high = 0.026
        artificial_noise_step = 0.005

        prior_noise_low = 1.0
        prior_noise_high = 1.0
        prior_noise_step = 0.1

        #Create parameter grid
        for artificial_noise in np.arange(artificial_noise_low, artificial_noise_high + artificial_noise_step*0.9, artificial_noise_step):
            for prior_noise in np.arange(prior_noise_low, prior_noise_high + prior_noise_step*0.9, prior_noise_step):
                kbnn_pars.append((x_train,y_train,x_test,y_test,layers,act_func,
                                artificial_noise,prior_noise,i))

        # MCMC
        mcmc_pars = []
        
        num_samples = 100
        data_noise = None

        prior_noise_low = 1.0
        prior_noise_high = 1.0
        prior_noise_step = 10

        # Create the parameter grid
        for prior_noise in np.arange(prior_noise_low, prior_noise_high + prior_noise_step*0.9, prior_noise_step):
            mcmc_pars.append((x_train,y_train,x_test,y_test,layers[1],prior_noise,
                                num_samples,data_noise,i))

        for j, parset in enumerate(progressive_pars, start=1):
            # Check wether run is already made
            if parset[9] is None:
                run_name = f"Progressive_NP{parset[8]}_{parset[6]}_{parset[7]}_L{'_LEARNED_'}_A{parset[10]:.4f}_P{parset[11]:.4f}_idx{i}"
            else:
                run_name = f"Progressive_NP{parset[8]}_{parset[6]}_{parset[7]}_L{parset[9]:.4f}_A{parset[10]:.4f}_P{parset[11]:.4f}_idx{i}"
            if os.path.exists(path_save + run_name + '.pkl'):
                print(f"Run {run_name} already exists. Skipping...")
                continue
            # Run
            print(f"Running Progressive parameter set {j}/{len(progressive_pars)}")
            result = rf.run_PROGRESSIVE(*parset,meas_guess=0.5) #DOCH WIEDER LOGNROMAL?
            if not result is None:
                with open(path_save+result["run_name"]+'.pkl', 'wb') as f:
                    pickle.dump(result, f)
            else:
                print(f"Progressive failed for {parset[0]}!")

        for j, parset in enumerate(kbnn_pars, start=1):
            # Check wether run is already made
            run_name = f"KBNN_A{parset[6]:.6f}_P{parset[7]:.4f}_idx{i}"
            if os.path.exists(path_save + run_name + '.pkl'):
                print(f"Run {run_name} already exists. Skipping...")
                continue
            # Run
            print(f"Running KBNN parameter set {j}/{len(kbnn_pars)}")
            result = rf.run_KBNN(*parset)
            if not result is None:
                with open(path_save+result["run_name"]+'.pkl', 'wb') as f:
                    pickle.dump(result, f)
            else:
                print(f"KBNN failed for {parset[0]}!")


        for j, parset in enumerate(mcmc_pars, start=1):
            # Check wether run is already made
            if parset[7] is None:
                run_name = f"MCMC_NS{parset[6]}_L{'_LEARNED_'}_P{parset[5]:.4f}_idx{i}"
            else:
                run_name = f"MCMC_NS{parset[6]}_L{parset[7]:.4f}_P{parset[5]:.4f}_idx{i}"
            if os.path.exists(path_save + run_name + '.pkl'):
                print(f"Run {run_name} already exists. Skipping...")
                continue
            # Run
            print(f"Running MCMC parameter set {j}/{len(mcmc_pars)} "+run_name)
            result = rf.run_MCMC(*parset)
            if not result is None:
                with open(path_save+result["run_name"]+'.pkl', 'wb') as f:
                    pickle.dump(result, f)
            else:
                print(f"MCMC failed for {parset[0]}!")

    #******Evaluation******
    #Generate a list of pkl files in the directory
    files = os.listdir(path_save)
    files = [f for f in files if (f.endswith(".pkl"))]
    files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(path_save, x)))

    if files == []:
        print("No files found")
        exit()

    #Evaluate and save
    results = []
    for file in files:
        results.append(hf.evaluate(path_save,file))

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
        combined_performance[i]["uce_mean"] = torch.mean(combined_performance[i]["uce"])
        combined_performance[i]["nll_mean"] = torch.mean(combined_performance[i]["nll"])
        combined_performance[i]["train_time_mean"] = torch.mean(combined_performance[i]["train_time"])
    # Sort by rmse_mean
    combined_performance = sorted(combined_performance, key=lambda x: x["rmse_mean"])

    # Print the first five candidates
    print("All candidates based on RMSE:")
    for candidate in combined_performance:
        print(f"Run Name: {candidate['run_name']}, RMSE: {candidate['rmse_mean'].item():.4f} ± {candidate['rmse'].std(dim=0).item():.4f}, UCE: {candidate['uce_mean'].item():.4f} ± {candidate['uce'].std(dim=0).item():.4f}, NLL: {candidate['nll_mean'].item():.4f} ± {candidate['nll'].std(dim=0).item():.4f}, Train Time: {candidate['train_time_mean'].item():.4f} ± {candidate['train_time'].std(dim=0).item():.4f}")
    # Save the results in a txt file
    with open(path_save + "results.txt", "w") as f:
        for candidate in combined_performance:
            f.write(f"Run Name: {candidate['run_name']}, RMSE: {candidate['rmse_mean'].item():.4f} ± {candidate['rmse'].std(dim=0).item():.4f}, UCE: {candidate['uce_mean'].item():.4f} ± {candidate['uce'].std(dim=0).item():.4f}, NLL: {candidate['nll_mean'].item():.4f} ± {candidate['nll'].std(dim=0).item():.4f}, Train Time: {candidate['train_time_mean'].item():.4f} ± {candidate['train_time'].std(dim=0).item():.4f}\n")

if run_yacht:
    print()
    print("Running Yacht Dataset...")
    print()
    #******Control Variables******
    torch.manual_seed(41)
    np.random.seed(41)

    # Path to save the data
    path_save = base_path + "yacht/"
    # Path to get the data from
    path_load = "datasets/"

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    #******Data Loading******	
    name = "dataset_yacht_data"
    with open(os.path.join(path_load, f"{name}.pkl"), "rb") as f:
        data_dict = pickle.load(f)
    X = data_dict["X"]
    Y = data_dict["Y"]
    x_scaler = data_dict["x_scaler"]
    y_scaler = data_dict["y_scaler"]

    # Convert to torch tensors
    x = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(Y, dtype=torch.float32)

    # Ensure y is 2D
    if y.ndimension() == 1:
        y = y.unsqueeze(1)

    print("Number of Features: ", x.shape[1])
    print("Number of Samples: ", x.shape[0])
    print("Number of Outputs: ", y.shape[0])

    #******Run environment******
    perm_list = []

    # Set the shuffled indices for each run beforehand so they stay the same
    for i in range(number_runs):
        perm = torch.randperm(x.size(0))
        perm_list.append(perm)

    for i in range(number_runs):
        #******Data Shuffling******
        indices = perm_list[i]
        x = x[indices]
        y = y[indices]

        #******Data Splitting******
        split_rate = 0.8
        split = int(split_rate * x.size(0))
        x_train = x[:split]
        y_train = y[:split]
        x_test = x[split:]
        y_test = y[split:]

        #******Architecture Parameters******
        # Define the parameters for my one hidden layer network (Relu Activated)
        hidden_units = 50
        layers = [x.shape[1],hidden_units,1]
        act_func = ["relu","linear"]

        #******Assemble Run Parameters******
        # PROGRESSIVE
        progressive_pars = []

        algo  = ["progressive_lcd"]
        training_type = ["sequential"]
        num_particles = [25000]

        data_noise = None

        artificial_noise_low = 0.0
        artificial_noise_high = 0.0
        artificial_noise_step = 0.05

        prior_noise_low = 1.0
        prior_noise_high = 1.0
        prior_noise_step = 0.5

        # Create the parameter grid
        for algo_type in algo:
            for num_part in num_particles:
                for train_type in training_type:
                    for artificial_noise in np.arange(artificial_noise_low, artificial_noise_high + artificial_noise_step, artificial_noise_step):
                        for prior_noise in np.arange(prior_noise_low, prior_noise_high + prior_noise_step, prior_noise_step):
                            progressive_pars.append((x_train,y_train,x_test,y_test,layers,act_func,
                                                algo_type,train_type,num_part,data_noise,artificial_noise,prior_noise,i))

        # KBNN
        kbnn_pars = []

        artificial_noise_low = 0.021
        artificial_noise_high = 0.021
        artificial_noise_step = 10

        prior_noise_low = 1.0
        prior_noise_high = 1.0
        prior_noise_step = 10

        #Create parameter grid
        for artificial_noise in np.arange(artificial_noise_low, artificial_noise_high + artificial_noise_step, artificial_noise_step):
            for prior_noise in np.arange(prior_noise_low, prior_noise_high + prior_noise_step, prior_noise_step):
                kbnn_pars.append((x_train,y_train,x_test,y_test,layers,act_func,
                                artificial_noise,prior_noise,i))
                
        # MCMC
        mcmc_pars = []
        
        num_samples = 100
        data_noise = None

        prior_noise_low = 1.0
        prior_noise_high = 1.0
        prior_noise_step = 10

        # Create the parameter grid
        for prior_noise in np.arange(prior_noise_low, prior_noise_high + prior_noise_step*0.9, prior_noise_step):
            mcmc_pars.append((x_train,y_train,x_test,y_test,layers[1],prior_noise,
                                num_samples,data_noise,i))

        for j, parset in enumerate(progressive_pars, start=1):
            # Check wether run is already made
            if parset[9] is None:
                run_name = f"Progressive_NP{parset[8]}_{parset[6]}_{parset[7]}_L{'_LEARNED_'}_A{parset[10]:.4f}_P{parset[11]:.4f}_idx{i}"
            else:
                run_name = f"Progressive_NP{parset[8]}_{parset[6]}_{parset[7]}_L{parset[9]:.4f}_A{parset[10]:.4f}_P{parset[11]:.4f}_idx{i}"
            if os.path.exists(path_save + run_name + '.pkl'):
                print(f"Run {run_name} already exists. Skipping...")
                continue
            # Run
            print(f"Running Progressive parameter set {j}/{len(progressive_pars)}")
            result = rf.run_PROGRESSIVE(*parset,meas_guess=1.0)
            if not result is None:
                with open(path_save+result["run_name"]+'.pkl', 'wb') as f:
                    pickle.dump(result, f)
            else:
                print(f"Progressive failed for {parset[0]}!")

        for j, parset in enumerate(kbnn_pars, start=1):
            # Check wether run is already made
            run_name = f"KBNN_A{parset[6]:.6f}_P{parset[7]:.4f}_idx{i}"
            if os.path.exists(path_save + run_name + '.pkl'):
                print(f"Run {run_name} already exists. Skipping...")
                continue
            # Run
            print(f"Running KBNN parameter set {j}/{len(kbnn_pars)}")
            result = rf.run_KBNN(*parset)
            if not result is None:
                with open(path_save+result["run_name"]+'.pkl', 'wb') as f:
                    pickle.dump(result, f)
            else:
                print(f"KBNN failed for {parset[0]}!")

        for j, parset in enumerate(mcmc_pars, start=1):
            # Check wether run is already made
            if parset[7] is None:
                run_name = f"MCMC_NS{parset[6]}_L{'_LEARNED_'}_P{parset[5]:.4f}_idx{i}"
            else:
                run_name = f"MCMC_NS{parset[6]}_L{parset[7]:.4f}_P{parset[5]:.4f}_idx{i}"
            if os.path.exists(path_save + run_name + '.pkl') :
                print(f"Run {run_name} already exists. Skipping...")
                continue
            # Run
            print(f"Running MCMC parameter set {j}/{len(mcmc_pars)} "+run_name)
            result = rf.run_MCMC(*parset)
            if not result is None:
                with open(path_save+result["run_name"]+'.pkl', 'wb') as f:
                    pickle.dump(result, f)
            else:
                print(f"MCMC failed for {parset[0]}!")

    #******Evaluation******
    #Generate a list of pkl files in the directory
    files = os.listdir(path_save)
    files = [f for f in files if (f.endswith(".pkl"))]
    files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(path_save, x)))

    if files == []:
        print("No files found")
        exit()

    #Evaluate and save
    results = []
    for file in files:
        results.append(hf.evaluate(path_save,file))

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
        combined_performance[i]["uce_mean"] = torch.mean(combined_performance[i]["uce"])
        combined_performance[i]["nll_mean"] = torch.mean(combined_performance[i]["nll"])
        combined_performance[i]["train_time_mean"] = torch.mean(combined_performance[i]["train_time"])
    # Sort by rmse_mean
    combined_performance = sorted(combined_performance, key=lambda x: x["rmse_mean"])

    # Print the first five candidates
    print("All candidates based on RMSE:")
    for candidate in combined_performance:
        print(f"Run Name: {candidate['run_name']}, RMSE: {candidate['rmse_mean'].item():.4f} ± {candidate['rmse'].std(dim=0).item():.4f}, UCE: {candidate['uce_mean'].item():.4f} ± {candidate['uce'].std(dim=0).item():.4f}, NLL: {candidate['nll_mean'].item():.4f} ± {candidate['nll'].std(dim=0).item():.4f}, Train Time: {candidate['train_time_mean'].item():.4f} ± {candidate['train_time'].std(dim=0).item():.4f}")
    # Save the results in a txt file
    with open(path_save + "results.txt", "w") as f:
        for candidate in combined_performance:
            f.write(f"Run Name: {candidate['run_name']}, RMSE: {candidate['rmse_mean'].item():.4f} ± {candidate['rmse'].std(dim=0).item():.4f}, UCE: {candidate['uce_mean'].item():.4f} ± {candidate['uce'].std(dim=0).item():.4f}, NLL: {candidate['nll_mean'].item():.4f} ± {candidate['nll'].std(dim=0).item():.4f}, Train Time: {candidate['train_time_mean'].item():.4f} ± {candidate['train_time'].std(dim=0).item():.4f}\n")

if run_power:
    print()
    print("Running Power Dataset...")
    print()
    #******Control Variables******
    torch.manual_seed(41)
    np.random.seed(41)

    # Path to save the data
    path_save = base_path + "power/"
    # Path to get the data from
    path_load = "datasets/"

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    #******Data Loading******	
    name = "dataset_power_data"
    with open(os.path.join(path_load, f"{name}.pkl"), "rb") as f:
        data_dict = pickle.load(f)
    X = data_dict["X"]
    Y = data_dict["Y"]
    x_scaler = data_dict["x_scaler"]
    y_scaler = data_dict["y_scaler"]

    # Convert to torch tensors
    x = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(Y, dtype=torch.float32)

    # Ensure y is 2D
    if y.ndimension() == 1:
        y = y.unsqueeze(1)

    print("Number of Features: ", x.shape[1])
    print("Number of Samples: ", x.shape[0])
    print("Number of Outputs: ", y.shape[0])

    #******Run environment******
    perm_list = []
    # Set the shuffled indices for each run beforehand so they stay the same
    for i in range(number_runs):
        perm = torch.randperm(x.size(0))
        perm_list.append(perm)

    for i in range(number_runs):
        #******Data Shuffling******
        indices = perm_list[i]
        x = x[indices]
        y = y[indices]

        #******Data Splitting******
        split_rate = 0.8
        split = int(split_rate * x.size(0))
        x_train = x[:split]
        y_train = y[:split]
        x_test = x[split:]
        y_test = y[split:]

        #******Architecture Parameters******
        # Define the parameters for my one hidden layer network (Relu Activated)
        hidden_units = 50
        layers = [x.shape[1],hidden_units,1]
        act_func = ["relu","linear"]

        #******Assemble Run Parameters******
        # PROGRESSIVE
        progressive_pars = []

        algo  = ["progressive"]
        training_type = ["sequential"]
        num_particles = [25000]

        data_noise = None

        artificial_noise_low = 0.0
        artificial_noise_high = 0.0
        artificial_noise_step = 0.05

        prior_noise_low = 1.0
        prior_noise_high = 1.0
        prior_noise_step = 0.5

        # Create the parameter grid
        for algo_type in algo:
            for num_part in num_particles:
                for train_type in training_type:
                    for artificial_noise in np.arange(artificial_noise_low, artificial_noise_high + artificial_noise_step, artificial_noise_step):
                        for prior_noise in np.arange(prior_noise_low, prior_noise_high + prior_noise_step, prior_noise_step):
                            progressive_pars.append((x_train,y_train,x_test,y_test,layers,act_func,
                                                algo_type,train_type,num_part,data_noise,artificial_noise,prior_noise,i))
        # KBNN
        kbnn_pars = []

        artificial_noise_low = 0.005
        artificial_noise_high = 0.005
        artificial_noise_step = 10

        prior_noise_low = 1.0
        prior_noise_high = 1.0
        prior_noise_step = 10

        #Create parameter grid
        for artificial_noise in np.arange(artificial_noise_low, artificial_noise_high + artificial_noise_step*0.9, artificial_noise_step):
            for prior_noise in np.arange(prior_noise_low, prior_noise_high + prior_noise_step*0.9, prior_noise_step):
                kbnn_pars.append((x_train,y_train,x_test,y_test,layers,act_func,
                                artificial_noise,prior_noise,i))
        # MCMC
        mcmc_pars = []
        
        num_samples = 100
        data_noise = None

        prior_noise_low = 1.0
        prior_noise_high = 1.0
        prior_noise_step = 10

        # Create the parameter grid
        for prior_noise in np.arange(prior_noise_low, prior_noise_high + prior_noise_step*0.9, prior_noise_step):
            mcmc_pars.append((x_train,y_train,x_test,y_test,layers[1],prior_noise,
                                num_samples,data_noise,i))

        for j, parset in enumerate(progressive_pars, start=1):
            # Check wether run is already made
            if parset[9] is None:
                run_name = f"Progressive_NP{parset[8]}_{parset[6]}_{parset[7]}_L{'_LEARNED_'}_A{parset[10]:.4f}_P{parset[11]:.4f}_idx{i}"
            else:
                run_name = f"Progressive_NP{parset[8]}_{parset[6]}_{parset[7]}_L{parset[9]:.4f}_A{parset[10]:.4f}_P{parset[11]:.4f}_idx{i}"
            if os.path.exists(path_save + run_name + '.pkl') :
                print(f"Run {run_name} already exists. Skipping...")
                continue
            # Run
            print(f"Running Progressive parameter set {j}/{len(progressive_pars)}")
            result = rf.run_PROGRESSIVE(*parset,meas_guess=1.0)
            if not result is None:
                with open(path_save+result["run_name"]+'.pkl', 'wb') as f:
                    pickle.dump(result, f)
            else:
                print(f"Progressive failed for {parset[0]}!")

        for j, parset in enumerate(kbnn_pars, start=1):
            # Check wether run is already made
            run_name = f"KBNN_A{parset[6]:.6f}_P{parset[7]:.4f}_idx{i}"
            if os.path.exists(path_save + run_name + '.pkl'):
                print(f"Run {run_name} already exists. Skipping...")
                continue
            # Run
            print(f"Running KBNN parameter set {j}/{len(kbnn_pars)}")
            result = rf.run_KBNN(*parset)
            if not result is None:
                with open(path_save+result["run_name"]+'.pkl', 'wb') as f:
                    pickle.dump(result, f)
            else:
                print(f"KBNN failed for {parset[0]}!")

        for j, parset in enumerate(mcmc_pars, start=1):
            # Check wether run is already made
            if parset[7] is None:
                run_name = f"MCMC_NS{parset[6]}_L{'_LEARNED_'}_P{parset[5]:.4f}_idx{i}"
            else:
                run_name = f"MCMC_NS{parset[6]}_L{parset[7]:.4f}_P{parset[5]:.4f}_idx{i}"
            if os.path.exists(path_save + run_name + '.pkl'):
                print(f"Run {run_name} already exists. Skipping...")
                continue
            # Run
            print(f"Running MCMC parameter set {j}/{len(mcmc_pars)} "+run_name)
            result = rf.run_MCMC(*parset)
            if not result is None:
                with open(path_save+result["run_name"]+'.pkl', 'wb') as f:
                    pickle.dump(result, f)
            else:
                print(f"MCMC failed for {parset[0]}!")

    #******Evaluation******
    #Generate a list of pkl files in the directory
    files = os.listdir(path_save)
    files = [f for f in files if (f.endswith(".pkl"))]
    files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(path_save, x)))

    if files == []:
        print("No files found")
        exit()

    #Evaluate and save
    results = []
    for file in files:
        results.append(hf.evaluate(path_save,file))

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
        combined_performance[i]["uce_mean"] = torch.mean(combined_performance[i]["uce"])
        combined_performance[i]["nll_mean"] = torch.mean(combined_performance[i]["nll"])
        combined_performance[i]["train_time_mean"] = torch.mean(combined_performance[i]["train_time"])
    # Sort by rmse_mean
    combined_performance = sorted(combined_performance, key=lambda x: x["rmse_mean"])

    # Print the first five candidates
    print("All candidates based on RMSE:")
    for candidate in combined_performance:
        print(f"Run Name: {candidate['run_name']}, RMSE: {candidate['rmse_mean'].item():.4f} ± {candidate['rmse'].std(dim=0).item():.4f}, UCE: {candidate['uce_mean'].item():.4f} ± {candidate['uce'].std(dim=0).item():.4f}, NLL: {candidate['nll_mean'].item():.4f} ± {candidate['nll'].std(dim=0).item():.4f}, Train Time: {candidate['train_time_mean'].item():.4f} ± {candidate['train_time'].std(dim=0).item():.4f}")
    # Save the results in a txt file
    with open(path_save + "results.txt", "w") as f:
        for candidate in combined_performance:
            f.write(f"Run Name: {candidate['run_name']}, RMSE: {candidate['rmse_mean'].item():.4f} ± {candidate['rmse'].std(dim=0).item():.4f}, UCE: {candidate['uce_mean'].item():.4f} ± {candidate['uce'].std(dim=0).item():.4f}, NLL: {candidate['nll_mean'].item():.4f} ± {candidate['nll'].std(dim=0).item():.4f}, Train Time: {candidate['train_time_mean'].item():.4f} ± {candidate['train_time'].std(dim=0).item():.4f}\n")