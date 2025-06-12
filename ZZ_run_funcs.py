"""
UU_run_funcs.py

Contains generalized function calls for all the regression problems.
It is implemented for PROGRESSIVE, KBNN and MCMC

"""
import os
import sys
import time
import numpy as np
import torch
from models.PROGRESSIVE_NET import PROGRESSIVE
#from models.KBNN import Bayesian_Network_torch as KBNN
from models.MCMC import MCMC_NET
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive
import jax
import jax.numpy as jnp

#******Control Variables******
torch.manual_seed(41)
np.random.seed(41)
numpyro.set_host_device_count(10)   # Set the maximum number of cpu cores to be used in numpyro

def run_PROGRESSIVE(x_train:torch.tensor,y_train:torch.tensor,x_test:torch.tensor,y_test:torch.tensor,
                layers,act_func,
                algo,training_type, num_particles, data_noise, artificial_noise,prior_noise,idx,
                x_ext:torch.tensor = None,y_ext:torch.tensor = None, meas_guess = None):
    """
    Function to run the Progressive Net with the given parameters.
    """
    #Check wether the noise is specified or will be learned
    if data_noise is None:
        run_name = f"Progressive_NP{num_particles}_{algo}_{training_type}_L{'_LEARNED_'}_A{artificial_noise:.4f}_P{prior_noise:.4f}_idx{idx}"
    else:
        run_name = f"Progressive_NP{num_particles}_{algo}_{training_type}_L{data_noise:.4f}_A{artificial_noise:.4f}_P{prior_noise:.4f}_idx{idx}"
    #data_noise = None
    try:

        #Initialize
        model = PROGRESSIVE(layers,act_func,num_particles, algo,training_type, meas_variance=data_noise,artificial_variance=artificial_noise,prior_variance=prior_noise,lcd_path="big_particles/",use_cuda=True,meas_mean_guess=meas_guess)

        # Train
        start_train = time.time()
        ###
        model.train(x_train,y_train,training_type)
        ###
        end_train = time.time()
        train_time = end_train - start_train

        # Predict (On test)
        start_pred = time.time()
        ###
        y_pred = model.predict(x_test)
        ###
        end_pred = time.time()
        pred_time = end_pred - start_pred

        # Compute output
        mean_pred = y_pred.mean(dim=1)
        var_pred = y_pred.var(dim=1)

        # If exists, predict on extended data
        if x_ext is not None:
            y_ext_pred = model.predict(x_ext)
            ext_pred_mean = y_ext_pred.mean(dim=1)
            ext_pred_var = y_ext_pred.var(dim=1)
        else:
            ext_pred_mean = None
            ext_pred_var = None

        # Assemble Data
        data = {'run_name':run_name,
                'x_train':x_train,'y_train':y_train,'x_test':x_test,'y_test':y_test,
                'mean_pred':mean_pred,'var_pred':var_pred, 'x_ext':x_ext,'y_ext':y_ext,
                'ext_mean_pred':ext_pred_mean,'ext_var_pred':ext_pred_var,
                'train_time':train_time,'pred_time':pred_time,
                'particles':model.particles,'idx':idx}

        # Return data
        return data
    
    except Exception as e:
        print(e)
        print(run_name+" failed!")
        return None

def run_KBNN(x_train:torch.tensor,y_train:torch.tensor,x_test:torch.tensor,y_test:torch.tensor,
            layers,act_func,
            artificial_noise,prior_noise,idx,
            x_ext:torch.tensor = None,y_ext:torch.tensor = None):
    """
    Function to run the KBNN network with the given parameters.
    """
    print("KBNN code is not yet published.")
    return None

def run_MCMC(x_train:torch.tensor,y_train:torch.tensor,x_test:torch.tensor,y_test:torch.tensor,
            hidden,prior_noise,
            num_samples,data_noise,idx,
            x_ext:torch.tensor = None,y_ext:torch.tensor = None):
    """
    Function to run the MCMC network with the given parameters.
    """
    if data_noise is None:
        run_name = f"MCMC_NS{num_samples}_L{'_LEARNED_'}_P{prior_noise:.4f}_idx{idx}"
    else:
        run_name = f"MCMC_NS{num_samples}_L{data_noise:.4f}_P{prior_noise:.4f}_idx{idx}"
    try:

        x_train = jnp.array(x_train)
        y_train = jnp.array(y_train)
        x_test = jnp.array(x_test)
        y_test = jnp.array(y_test)
        if x_ext is not None:
            x_ext = jnp.array(x_ext)
            y_ext = jnp.array(y_ext)

        hidden_units = hidden

        # Initialize
        if len(x_train.shape) == 2:
            in_feat = x_train.shape[1]
        else:
            in_feat = 1
        model = MCMC_NET(in_feat,hidden_units,1,data_noise)
        nuts_kernel = NUTS(model.model)

        # Train
        start_train = time.time()
        ###
        mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=int(num_samples/2),num_chains=10,chain_method="parallel")
        print("Attention! Running MCMC with parallel chains Keep in mind when counting number of samples!")
        rng_key = jax.random.PRNGKey(1)
        mcmc.run(rng_key,x_train, y_train.squeeze())    # Attention! y_train has to be a 1D tensor???
        posterior_samples = mcmc.get_samples()

        ###
        end_train = time.time()
        train_time = end_train - start_train

        # Prediction Class
        predictive = Predictive(model.model, posterior_samples,return_sites=["obs"])

        # Predict (On test)
        start_pred = time.time()
        ###
        predictions= predictive(jax.random.PRNGKey(2),x_test)
        y_pred = predictions["obs"]
        y_pred = y_pred.T
        ###
        end_pred = time.time()
        pred_time = end_pred - start_pred

        # Compute output
        mean_pred = jnp.mean(y_pred,axis = 1)
        var_pred = jnp.var(y_pred,axis = 1)

        # If exists, predict on extended data
        if x_ext is not None:
            ext_pred = predictive(jax.random.PRNGKey(3),x_ext)
            ext_pred = ext_pred["obs"]
            ext_pred = ext_pred.T
            ext_pred_mean = jnp.mean(ext_pred,axis = 1)
            ext_pred_var = jnp.var(ext_pred,axis = 1)
        else:
            ext_pred_mean = None
            ext_pred_var = None

        x_train = torch.tensor(x_train)
        y_train = torch.tensor(y_train)
        x_test = torch.tensor(x_test)
        y_test = torch.tensor(y_test)
        y_pred = torch.tensor(y_pred)
        mean_pred = torch.tensor(mean_pred)
        var_pred = torch.tensor(var_pred)
        if x_ext is not None:
            x_ext = torch.tensor(x_ext)
            y_ext = torch.tensor(y_ext)
            ext_pred_mean = torch.tensor(ext_pred_mean)
            ext_pred_var = torch.tensor(ext_pred_var)

        # Assemble Data
        data = {'run_name':run_name,
                'x_train':x_train,'y_train':y_train,'x_test':x_test,'y_test':y_test,
                'mean_pred':mean_pred,'var_pred':var_pred, 'x_ext':x_ext,'y_ext':y_ext,
                'ext_mean_pred':ext_pred_mean,'ext_var_pred':ext_pred_var,
                'train_time':train_time,'pred_time':pred_time,
                'posterior_samples':posterior_samples,'idx':idx}

        # Save Data
        return data
    except Exception as e:
        print(e)
        print(run_name+" failed!")