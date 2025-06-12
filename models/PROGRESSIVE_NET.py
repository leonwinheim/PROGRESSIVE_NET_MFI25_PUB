"""
PROGRESSIVE_NET.py

This module containes the implementation of the "PROGRESSIVE NET", which is a Bayesian Neural Network (BNN) trained with a Progressive Gaussian Filter.
Developed by Leon Winheim and Uwe D. Hanebeck at the Intelligent Sensor-Actuator-Systems Group (ISAS), Karlsruhe Institute of Technology (KIT).
For questions and remarks, please contact leon.winheim@kit.edu

"""
import torch
import numpy as np
from tqdm import tqdm
from functools import partial
import pickle as pkl
import time
import os
import pandas as pd

#******Activation Functions******
def relu_handle(x:torch.Tensor):
    """
    ReLU activation function
    """
    return torch.max(x,torch.zeros_like(x))

def linear_handle(x:torch.Tensor):
    """
    Linear activation function
    """
    return x

def sigmoid_handle(x:torch.Tensor):
    """
    Sigmoid activation function
    """
    return 1/(1+torch.exp(-x))

#******Class Definition******
class PROGRESSIVE:
    """
    PROGRESSIVE NET is a Bayesian Neural Network (BNN) trained with a Progressive Gaussian Filter. For a deterministic input, it yields a predictive distribution over the output.
    The implementation is based on torch and uses CUDA acceleration if available.
    Apart from progressive implementations, a Bootstrap Particle Filter ist available as training mode.
    """
    
    def __init__(self, layers:list, act_func:list, num_particles:int, algorithm:str, mode:str,
                meas_variance = None,
                artificial_variance = 0.001,
                prior_variance = 1.0,
                meas_mean_guess = None,
                lcd_path = None,
                use_cuda= True):
        """
        Initializes an instance of the PROGRESSIVE NET with given parameters.

        Parameters:
        - layers (list): List of integers representing the number of neurons in each layer.
        - act_func (list): List of activation functions for each layer. Contains Strings, will be converted to torch functions.
        - num_particles (int): Number of particles (or samples) used for the filter step.
        - algorithm (str): Algorithm to be used for training. Valid options: "bootstrap", "progressive", "progressive_lcd".
        - mode (str): Regarding processing type. Can be "sequential" or "minibatch". In the "minibatch"-case, batchsize will be specified in a later call to the `train` method.
        - meas_variance (float or list, optional): Measurement variance for the likelihood used in the filter step. Defaults to None, which means it will be estimated from the data. If multivariate output, it must be a list with the diagonal elements.
        - artificial_variance (float, optional): Artificial noise to be added to the system dynamics (Used only with bootstrap now). Defaults to 0.001.
        - prior_variance (float, optional): Variance of the prior distribution. Defaults to 1.0.
        - meas_mean_guess (float, optional): Initial guess for the measurement variance if it is should be inferred from the data. Defaults to None
        - lcd_path (str, optional): Path to the deterministic samples. Defaults to None, in which case it has to be set elsewhere
        - use_cuda (bool, optional): Whether to use CUDA for acceleration. Defaults to True.

        """
        self.act_func = act_func.copy()
        self.layers = layers
        self.num_layers = len(layers)
        self.num_particles = num_particles
        self.algorithm = algorithm
        self.mode = mode
        self.meas_noise = meas_variance
        self.meas_mean_guess = meas_mean_guess
        self.artificial_variance = artificial_variance
        self.prior_variance = prior_variance
        self.use_cuda = use_cuda

        self.valid_algorithms = ["bootstrap", "progressive", "progressive_lcd"]
        self.valid_modes = ["sequential", "minibatch"]

        # Set the LCD path 
        self.set_lcd_path(lcd_path)

        # Start the initialization procedure
        self.init_network()

        # Define some constants so I dont need to compuite in place
        self.pi_tensor = torch.tensor(np.pi)

        # Give out a warning
        if (self.meas_noise is not None) and (meas_mean_guess is not None):
            print("Warning: Measurement noise is set to a fixed value. This will not be estimated from the data.")
        elif (self.meas_noise is None) and (meas_mean_guess is None):
           raise ValueError("Measurement noise is not set. Please provide a value or a guess for the measurement noise prior.")

    def print_info(self):
        """
        Print network information
        """
        print("******Network-Information******")
        print("Algorithm: ",self.algorithm)
        print("Mode: ",self.mode)
        print("Number of Layers: ",self.num_layers)
        print("Layer Sizes: ",self.layers)
        print("Activation Functions: ",self.act_func)
        print("Number of Particles: ",self.num_particles)
        print("*******************************")
    
    @torch.no_grad()
    def init_network(self):
        """
        Initialize the network weights and everything else.
        """
        # Check for Validity of shapes
        # Scalar output case
        if self.layers[-1] == 1:
            # Ensure self.meas_noise is either a scalar or a list of length 1
            if isinstance(self.meas_noise, list):
                assert len(self.meas_noise) == 1, "meas_noise should be a scalar float or a list of length 1 when the last layer entry is 1."
                self.meas_noise = self.meas_noise[0]
            elif self.meas_noise is None:
                pass
            else:
                assert isinstance(self.meas_noise, (float)),  "meas_noise should be a scalar float when the last layer entry is 1."
        # Multivariate Output case
        else:
            # Ensure self.meas_noise is a list of length equal to the last layer entry
            if isinstance(self.meas_noise, list):
                assert len(self.meas_noise) == self.layers[-1], "meas_noise should be a list of length equal to the last layer entry."
            elif self.meas_noise is None:
                pass
            else:
                raise ValueError("meas_noise should be a list when the output dimension is not 1.")
        
        #Convert self.meas_noise to a tensor
        if self.meas_noise is None:
            # In this case, the measurement noise will be estimated alongside with the weights
            pass
        elif isinstance(self.meas_noise, list):
            self.meas_noise = torch.tensor(self.meas_noise, dtype=torch.float32)
            self.meas_noise = torch.diag(self.meas_noise.flatten())                 # Convert the tensor into a diagonal matrix (no covariance entries, just variances)
            self.inv_meas_noise = torch.linalg.inv(self.meas_noise)                 # Invert it for later use in mahalanobins distance
        else:
            #Is this even neccesary?
            self.meas_noise = torch.tensor([[self.meas_noise]],dtype=torch.float32) # Convert the List into a tensor
            self.meas_noise = torch.diag(self.meas_noise.flatten())                 # Convert the tensor into a diagonal matrix (no covariance entries, just variances)
            self.inv_meas_noise = torch.linalg.inv(self.meas_noise)                 # Invert it for later use in mahalanobins distance
       
        # Compute number of parameters needed (weights + biases) for the whole network
        self.num_weights = 0
        for i in range(self.num_layers-1):
            self.num_weights += (self.layers[i]+1)*self.layers[i+1]
        
        print(f"This network will have {self.num_weights} weight parameters")

        # Initialize weights and biases with particles, last two columns are weight and likelihood
        ####
        # In every row of self.particles we get the following structure
        # [Layer 1: (n*m x 1) Layer 2:(n*m x 1) ....Layer l:(n*m x 1), weight, likelihood]
        # (n is neuron count) and (m is neuron count in layer before plus bias) in respective layer, so a "particle" (a row) is 
        # the weights flattend and concatenated.
        # !!One Particle is a row up to the last two columns (or up to the measurement noise + last two columns)!!
        ####
        if self.algorithm =="progressive_lcd":
            # Call the function to read a particle file
            self.lcd = self.load_lcd(self.num_weights, self.num_particles)

            ########
            #Gaussian Prior (LCD-Sampled)
            print("Gaussian prior generated")
            mean = torch.randn(self.num_weights)                                                                    #PRIOR MEAN
            # Set Bias-means to zero
            end_before = 0
            for  i in range(self.num_layers-1):
                mean[end_before+self.layers[i+1]*(self.layers[i]):end_before+self.layers[i+1]*(self.layers[i]+1)] = 0.0      
                end_before += self.layers[i+1]*(self.layers[i]+1)                            
            cov = torch.eye(self.num_weights)*self.prior_variance                                                   #PRIOR COVARIANCE MATRIX
            self.lcd_prior = self.scale_lcd(mean,cov)
            self.particles = self.lcd_prior
            ########

            ########
            #Uniform Prior 
            # print("Uniform prior generated")
            # low = -0.5 * self.prior_variance
            # high = 0.5 * self.prior_variance
            # self.particles = torch.empty((self.num_particles, self.num_weights)).uniform_(low, high)
            #######
            
            # Prepare estimation of measurement noise
            if self.meas_noise is None:
                # Append columns for every output neuron for the measurement noise from a Gaussian Distribution with mean 1
                # Sample measurement noise for each output neuron from a positive distribution (e.g., LogNormal or uniform)
                noise_samples = torch.distributions.LogNormal(self.meas_mean_guess,1.0).sample((self.num_particles, self.layers[-1]))
                #noise_samples = torch.distributions.Uniform(0.5*self.meas_mean_guess, 1.5*self.meas_mean_guess).sample((self.num_particles, self.layers[-1]))
                # Add a 1/N weight column to samples
                noise_samples = torch.cat((noise_samples, torch.ones(self.num_particles, 1)/self.num_particles), dim=1)
                self.noise_samples = noise_samples
            

            # Append the last two columns for likelihood and weights
            self.particles = torch.cat((self.particles, torch.zeros(self.num_particles, 2)), dim=1)
            self.particles[:,-1] = 0.0                          # Set likelihood to zero
            self.particles[:,-2] = 1.0/self.particles.shape[0]  # Set uniform weights

        else:
            ########
            # #Uniform Prior 
            # print("Uniform prior generated")
            # low = -0.5 * self.prior_variance
            # high = 0.5 * self.prior_variance
            # self.particles = torch.empty((self.num_particles, self.num_weights)).uniform_(low, high)
            ########

            ########
            # Gaussian Prior (Random Sampled)
            print("Gaussian prior generated")
            mean = torch.randn(self.num_weights)     
            # Set Bias-means to zero
            end_before = 0
            for  i in range(self.num_layers-1):
                mean[end_before+self.layers[i+1]*(self.layers[i]):end_before+self.layers[i+1]*(self.layers[i]+1)] = 0.0      
                end_before += self.layers[i+1]*(self.layers[i]+1)                                                                  #PRIOR MEAN
            cov = torch.eye(self.num_weights)*self.prior_variance                                                  #PRIOR COVARIANCE MATRIX
            self.particles = torch.distributions.MultivariateNormal(mean, cov).sample((self.num_particles,))       #PRIOR DISTRIBUTION
            #######

            # Prepare estimation of measurement noise
            if self.meas_noise is None:
                # Append columns for every output neuron for the measurement noise from a Gaussian Distribution with mean 1
                # Sample measurement noise for each output neuron from a positive distribution (e.g., LogNormal or Uniform)
                noise_samples = torch.distributions.LogNormal(self.meas_mean_guess, 1.0).sample((self.num_particles, self.layers[-1]))
                #noise_samples = torch.distributions.Uniform(0.5*self.meas_mean_guess, 1.5*self.meas_mean_guess).sample((self.num_particles, self.layers[-1]))
                # Add a 1/N weight column to samples
                noise_samples = torch.cat((noise_samples, torch.ones(self.num_particles, 1)/self.num_particles), dim=1)
                self.noise_samples = noise_samples

            # Append the last two columns for likelihood and weights
            self.particles = torch.cat((self.particles, torch.zeros(self.num_particles, 2)), dim=1)
            self.particles[:,-1] = 0.0                          # Set likelihood to zero
            self.particles[:,-2] = 1.0/self.particles.shape[0]  # Set uniform weights


        # Translate the Activation-Func-list to a list of function handles
        for i in range(len(self.act_func)):
            if self.act_func[i] == "relu":
                self.act_func[i] = relu_handle
            elif self.act_func[i] == "linear":
                self.act_func[i] = linear_handle
            elif self.act_func[i] == "sigmoid":
                self.act_func[i] = sigmoid_handle
            else:
                raise NotImplementedError(f"Desired Implementation not yet implemented: {self.act_func[i]}")
        
        # Check the validity of network mode
        if self.mode not in self.valid_modes:
            raise ValueError("Invalid mode. Please choose one of the following: ",self.valid_modes)
        
        # Check the validity of algorithm
        if self.algorithm not in self.valid_algorithms:
            raise ValueError("Invalid algorithm. Please choose one of the following: ",self.valid_algorithms)

    def reset_particles(self,particle_set:torch.Tensor):
        """
        Reset the particles to a given set of particles. This is used for loading particles from a file or manually.
        """
        assert particle_set.shape[1] == self.num_weights+2, "The given particle set does not match the network architecture."
        assert particle_set.shape[0] == self.num_particles, "The given particle set does not match the number of particles."

        self.particles = particle_set
    
    def save_particles(self, filename):
        """
        Save the current particle set to a file.
        """
        with open(filename+".pkl",'wb') as f:
            pkl.dump(self.particles, f)
    
    def load_particles(self, filename):
        """
        Load a particle set from a file.
        """
        with open(filename+".pkl",'rb') as f:
            data = pkl.load(f)
        self.reset_particles(data)
        if self.use_cuda:
            self.prepare_gpu()
        elif data.device.type == 'cuda':
            # If the data is on GPU but CUDA is not available, move it to CPU
            self.particles = data.cpu()
            print("Data was on GPU, but CUDA is not selected in initialization. Moving to CPU.")

    def set_lcd_path(self, path = None):
        """
        Set the path for loading the LCD samples. If None, the default path is used.
        """
        if path is None:
            self.lcd_path = "big_particles/"
        else:
            self.lcd_path = path

    def load_lcd(self, num_weights, num_particles):
        """
        Function to read  a sample file of lcd sample positions generated from the MATLAB CodeOcean file. 
        Original naming convention from MATLAB implementation
        """
        # Set a directory from where to load the pre-generated samples
        sample_path = self.lcd_path

        # List all files in the sample_path directory
        files = os.listdir(sample_path)

        # Filter for CSV files
        csv_files = [file for file in files if file.endswith('.csv')]

        # Check the name of the files to contain the following strings
        dim_str = "dim="+str(num_weights)+"_"
        part_str = "L="+str(num_particles)+".csv"

        # Filter for files that contain the specified strings
        filtered_files = [file for file in csv_files if dim_str in file and part_str in file]

        # Check if any files were found
        if not filtered_files:
            raise FileNotFoundError(f"No files found in {sample_path} matching the criteria {dim_str} and {part_str}.")
        # Ensure only one file matches the criteria
        if len(filtered_files) > 1:
            raise ValueError(f"Multiple files found in {sample_path} matching the criteria {dim_str} and {part_str}. Please ensure uniqueness.")

        # Load the unique file
        file_path = os.path.join(sample_path, filtered_files[0])

        # Read in and convert
        lcd_samples = torch.tensor(pd.read_csv(file_path, header=None).to_numpy(dtype=np.float32))

        # Check the shape of the loaded samples
        if lcd_samples.shape != (num_particles, num_weights):
            raise ValueError(f"The loaded LCD samples have shape {lcd_samples.shape}, but expected ({num_particles}, {num_weights}).")

        print("Loaded LCD Samples!")

        return lcd_samples
    
    @torch.no_grad()
    def scale_lcd(self, mean, cov):
        """
        Transforms a standard normal distribution to a normal distribution with given mean and covariance using magic
        """
        # Compute the Cholesky decomposition of the covariance matrix
        #A = torch.linalg.cholesky(cov+ torch.eye(cov.size(0)).to(cov.device) * 1e-5)  # Add a small value to the diagonal for numerical stability

        # Singular Value Decomposition (Not yet tried)
        # U, S, _ = torch.linalg.svd(cov)
        # # Take square root of the singular values
        # S = torch.diag(torch.sqrt(S))
        # # Compute the transformation matrix
        # A = U @ S

        # Egendecomposition (for numeric stability, though slower)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals_clipped = torch.clamp(eigvals, min=1e-6)
        A = eigvecs @ torch.diag(torch.sqrt(eigvals_clipped))

        # Scale the Samples based on presaved LCD samples read in before
        temp_lcd = self.lcd @ A.T 

        # Shift the Samples
        temp_lcd += mean

        return temp_lcd
    
    @torch.no_grad()
    def resample_noise(self):
        """Centralized Resampling function for the learnable noise case"""
        # Compute parameters
        #####If Lognormal is used for the noise
        log_pos = torch.log(self.noise_samples[:,0])
        loc_noise = torch.sum(log_pos * self.noise_samples[:,1])/ torch.sum(self.noise_samples[:,1])
        var_noise = torch.sum(self.noise_samples[:,1]*(log_pos-loc_noise)**2)/ torch.sum(self.noise_samples[:,1])
        #####If uniform is used for the noise
        #mean_noise = torch.sum(self.noise_samples[:,0] * self.noise_samples[:,1])/ torch.sum(self.noise_samples[:,1])
        #std_noise = torch.sqrt(torch.sum(self.noise_samples[:,1]*(self.noise_samples[:,0]-mean_noise)**2)/ torch.sum(self.noise_samples[:,1]))

        #Resample
        #####If lognromal is used
        self.noise_samples[:,0] = torch.distributions.LogNormal(loc_noise, torch.sqrt(var_noise)).sample((self.num_particles,))
        #####If uniform is used
        #if mean_noise-std_noise < 0.0:
            # If the mean minus the standard deviation is negative, we set it to 0.0
            #self.noise_samples[:,0] = torch.distributions.Uniform(0.0, mean_noise+std_noise).sample((self.num_particles,))
        #else:
            #self.noise_samples[:,0] = torch.distributions.Uniform(mean_noise-std_noise, mean_noise+std_noise).sample((self.num_particles,))
        self.noise_samples[:,1] = 1.0/self.num_particles
    
    @torch.no_grad()
    def compute_weighted_moments(self):
        """
        Returns the weighted mean vector and covariance matrix of the particles.
        This is used for the resampling step in the progressive filter.
        """
        # Weighted mean
        mean = torch.sum(self.particles[:,:-2] * self.particles[:,-2].unsqueeze(1), dim=0) / torch.sum(self.particles[:,-2])

        # Weighted covariance (Must be transposed due to torch.cov definition)
        cov = torch.cov(self.particles[:,:-2].T, aweights=self.particles[:,-2])

        return mean, cov
    
    def prepare_gpu(self):
        """
        Prepare the data for gpu-computation. This is used for CUDA support.
        """
        if torch.cuda.is_available():
            self.particles = self.particles.cuda()
            print("GPU support enabled.")
            # Move relevant tensors to GPU
            self.particles = self.particles.cuda()
            if not self.meas_noise is None:
                self.meas_noise = self.meas_noise.cuda()
                self.inv_meas_noise = self.inv_meas_noise.cuda()
            else:
                self.noise_samples= self.noise_samples.cuda()
            self.pi_tensor = self.pi_tensor.cuda()
            if self.algorithm == "progressive_lcd":
                self.lcd = self.lcd.cuda()

        else:
            print("CUDA not available. Buy a big fat GPU or set 'use_cuda=False' in training() function! Exiting...")
            exit()
    
    @torch.no_grad()
    def train(self,x,y,batchsize=10, adaptive_step=True, manual_step=0.1):
        """
        Method to call the desired training procedure. Algorithm and Mode are already set in constructor.

        Parameters:
        - x (torch.Tensor): Input features, shape (num_samples, input_dim).
        - y (torch.Tensor): Output targets, shape (num_samples, output_dim).
        - batchsize (int, optional): Batch size for minibatch training. Defaults to 10. Only used if specified before.
        - adaptive_step (bool, optional): Whether to use an adaptive step size for the progressive filters. Defaults to True.
        - manual_step (float, optional): Manual step size for the progressive filters. Defaults to 0.1. Only used if adaptive step is False.

        """
        
        # Expand output tensor if it is 1-D
        if len(y.shape) == 1:
            y = y.unsqueeze(-1)

        # Expand input tensor if it is 1-D
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)

        # Check wether the data dimensions are valid
        assert x.shape[0] == y.shape[0], "Features and outputs must have the same number of samples."
        assert x.shape[1] == self.layers[0], f"The input dimension does not match the first layer of the network. Expected {self.layers[0]}, got {x.shape[1]}."
        assert y.shape[1] == self.layers[-1], "The output dimension does not match the last layer of the network."
        assert type(manual_step) == float, "The manual step size must be a float."
        
        # Call the desired training procedure. "algorithm" corresponds to to the algorithm, "mode" to the training method regarding data intake
        # Also decide wether an adaptive stepsize is used for the progressive filters

        # Activate GPU support
        if self.use_cuda:
            self.prepare_gpu()
            x = x.cuda()
            y = y.cuda()

        if self.algorithm == "bootstrap":
            if self.mode == "minibatch":
                self.train_minibatch_bootstrap(x,y,batchsize)
            elif self.mode == "sequential":
                self.train_sequential_bootstrap(x,y)
        elif self.algorithm == "progressive":
            if adaptive_step:
                if self.mode == "minibatch":
                    self.train_minibatch_progressive_adaptive(x,y,batchsize)
                elif self.mode == "sequential":
                    self.train_sequential_progressive_adaptive(x,y)
            else:
                if self.mode == "minibatch":
                    self.train_minibatch_progressive(x,y,batchsize,manual_step)
                elif self.mode == "sequential":
                    self.train_sequential_progressive(x,y,manual_step)
        elif self.algorithm == "progressive_lcd":
            if adaptive_step:
                if self.mode == "minibatch":
                    self.train_minibatch_progressive_lcd_adaptive(x,y,batchsize)
                elif self.mode == "sequential":
                    self.train_sequential_progressive_lcd_adaptive(x,y)
            else:
                if self.mode == "minibatch":
                    self.train_minibatch_progressive_lcd(x,y,batchsize,manual_step)
                elif self.mode == "sequential":
                    self.train_sequential_progressive_lcd(x,y,manual_step)
        else:
            raise ValueError("Invalid mode. I should have caught this earlier")
        
    @torch.no_grad()
    def train_sequential_bootstrap(self,x,y):
        """
        Train the network with the given data !sequentially!. 
        Use the Bootstrap filter.
        """
        if self.meas_noise is None:
            raise ValueError("Learnable Noise not supported for Bootstrap Filter.")
        
        print("*****Started Training (Bootstrap Filter, Sequential Method)*****")
        start = time.time()

        # Iterate through data points
        for pt in tqdm(range(x.shape[0]),desc="Training (Sequential)"):
            # Chose training point and make it (1,d)-Dimensional 
            x_seq = x[pt].unsqueeze(0)
            y_seq = y[pt].unsqueeze(0)
            # Assertion statements
            assert x_seq.shape[1] == self.layers[0], f"The input dimension does not match the first layer of the network. Expected {self.layers[0]}, got {x_seq.shape[0]}."

            #Call Forward Function on all particles  (vectorized)
            _,log_likelihood=self.forward(x_seq,y_seq)

            # Add Result to log Likelihood
            self.particles[:, -1] =+log_likelihood

            # Perform the filtering step
            self.filter_step_bootstrap()

        end = time.time()

        print("*****Training Finished after ",np.round(end-start,2)," seconds*****")

    @torch.no_grad()
    def train_minibatch_bootstrap(self,x,y,batchsize):
        """
        Train the network with the given data !mini-batchwise!.
        Use the Bootstrap filter.
        """
        if self.meas_noise is None:
            raise ValueError("Learnable Noise not supported for Bootstrap Filter.")
    
        print("*****Started Training (Bootstrap Filter, Minibatch Method)*****")
        start = time.time()

        # Save original data
        x_orig = x.clone()
        y_orig = y.clone()

        # Compute number of batches (x_orig has the shape n x d with n training pts)
        batchnum = int(x_orig.shape[0]/batchsize)

        #Training Sequence in mini-batches
        for i in tqdm(range(batchnum),desc="Mini Batch"):
            # Subsample the big batch
            x = x_orig[i*int(x_orig.shape[0]/batchnum):(i+1)*int(x_orig.shape[0]/batchnum)]
            y = y_orig[i*int(y_orig.shape[0]/batchnum):(i+1)*int(y_orig.shape[0]/batchnum)]

            # Iterate through data points
            for pt in range(x.shape[0]):
                # Chose training point and make it (1,d)-Dimensional 
                x_seq = x[pt].unsqueeze(0)
                y_seq = y[pt].unsqueeze(0)

                #Call Forward Function on all particles  (vectorized)
                _,log_likelihood=self.forward(x_seq,y_seq)

                # Add Result to log Likelihood
                self.particles[:, -1] =+log_likelihood

            # Perform the filtering step (after a mini batch instead of every data point)
            self.filter_step_bootstrap()

        end = time.time()

        print("*****Training Finished after ",np.round(end-start,2)," seconds*****")

    @torch.no_grad()
    def train_sequential_progressive(self,x,y,prog_step):
        """
        Train the network with the given data !sequeniatlly!. 
        This version uses Progressive Gauss Filter and random gaussian samples for the resampling.
        """
        print("*****Started Training (Progressive Gauss Filter, Random Sampled, Sequential Method)*****")
        start = time.time()

        # Specify progression parameters
        prog_steps = 1.0/prog_step  #Compute the number of steps based on the manual step size
        gamma_inc = 1.0/prog_steps

        # Iterate through data points
        for pt in tqdm(range(x.shape[0]),desc="Training (Sequential)"):

            # Chose training point and make it (1,d)-Dimensional 
            x_seq = x[pt].unsqueeze(0)
            y_seq = y[pt].unsqueeze(0)

            #Progression: Iterate over gamma
            for i in range(1,prog_steps+1):
                # Set temporary gamma (progression exponent)
                gamma = i*gamma_inc

                #Call Forward Function on all particles  (vectorized)
                _,log_likelihood=self.forward(x_seq,y_seq)

                # Add Result to log Likelihood
                self.particles[:, -1] =+log_likelihood

                # Shift the log-likeihoods into the positive by adding the minimum value (stability)
                max_val = torch.max(self.particles[:,-1])
                self.particles[:,-1] = self.particles[:,-1] - max_val

                # Exponentiate the log-likelihoods
                self.particles[:,-1] = torch.exp(self.particles[:,-1])    

                # Compute the current progression state of the likelihood values
                temp_likelihood = self.particles[:,-1]**gamma_inc

                # Perform the weighting (elementwise)
                self.particles[:,-2] = self.particles[:,-2]*temp_likelihood

                # Normalize the weights
                self.particles[:,-2] = self.particles[:,-2]/torch.sum(self.particles[:,-2])
                
                # Apply the weights to the noise shape as well if it is subject to learning, then resample
                if self.meas_noise is None:
                    self.noise_samples[:,1] =  self.particles[:,-2]
                    self.resample_noise()

                # Compute moments for resampling
                mean, cov = self.compute_weighted_moments()
                cov += torch.eye(cov.size(0), device=self.particles.device) * 1e-6  # Add a small value to the diagonal for numerical stability
            
                try:
                    # Resample particles, drawn from a gaussian with computed moments and reset the samples
                    self.particles[:,:-2] = torch.distributions.MultivariateNormal(mean, cov).sample((self.num_particles,))
                    self.particles[:,-2] = 1.0/self.num_particles
                    
                except Exception as e:
                    print(f"Resampling failed with error: {e}")
                    print(f"Progressive step: {i}, gamma: {gamma}")
                    eigvals = torch.linalg.eigvalsh(cov)
                    print("Min Eigenvalue",eigvals.min())  # Should be > 0
                    print("Min diagonal element:", torch.min(torch.diag(cov)))
                    print("Max diagonal element:", torch.max(torch.diag(cov)))
                    raise ValueError("Resampling failed. Check the covariance matrix.")

                # Set likelihood to 0
                self.particles[:,-1] = 0.0

            #Check wether the complete likelihood was used in the end
            assert gamma == 1.0, "The progression loop did not reach the end."

        end = time.time()

        if self.meas_noise is None:
            print("*****Training Finished after ",np.round(end-start,2),f" seconds. Estimated noise: {self.noise_samples[:,0].mean():.2f}+- {self.noise_samples[:,0].std():.2f}*****")
        else:
            print("*****Training Finished after ",np.round(end-start,2)," seconds.*****")

    @torch.no_grad()
    def train_sequential_progressive_adaptive(self,x,y):
        """
        Train the network with the given data !sequeniatlly!.
        This version uses Progressive Gaussian Filtering with random gaussian samples for the resampling.
        This version uses a automatically determined stepsize based on Steinbring, Hanebeck Publication
        """
        print("*****Started Training (Progressive Gauss Filter Random Sampled, Sequential Method)*****")
        start = time.time()

        # Specify progression parameters
        log_M = torch.log(torch.tensor(self.num_particles))

        # Iterate through data points
        for pt in tqdm(range(x.shape[0]),desc="Training (Sequential)"):

            # Chose training point and make it (1,d)-Dimensional 
            x_seq = x[pt].unsqueeze(0)
            y_seq = y[pt].unsqueeze(0)

            # Set initial gamma
            gamma = 0.0

            #Progression: Iterate over gamma
            while gamma < 1.0:

                #Call Forward Function on all particles  (vectorized)
                _,log_likelihood=self.forward(x_seq,y_seq)

                # Add Result to log Likelihood
                self.particles[:, -1] =+log_likelihood

                # Shift the log-likeihoods into the positive by adding the minimum value (stability)
                max_val = torch.max(self.particles[:,-1])
                self.particles[:,-1] = self.particles[:,-1] - max_val

                #Compute the increment
                min_like = torch.min(self.particles[:,-1])
                max_like = torch.max(self.particles[:,-1])
                
                gamma_inc = -log_M/(min_like-max_like)

                # Ensure gamma_inc is not negative, min gamma_inc is adjustable
                if gamma_inc < 0.0: 
                    gamma_inc = 0.1
                    
                # Update gamma
                if gamma + gamma_inc > 1.0:
                    gamma_inc = 1.0 - gamma
                gamma += gamma_inc

                # Compute the current progression state of the likelihood values
                self.particles[:,-1] = torch.exp(self.particles[:,-1]*gamma_inc)

                # Perform the weighting (elementwise)
                self.particles[:,-2] = self.particles[:,-2]*self.particles[:,-1]

                # Normalize the weights
                self.particles[:,-2] = self.particles[:,-2]/torch.sum(self.particles[:,-2])
                
                # Apply the weights to the noise shape as well if it is subject to learning, then resample
                if self.meas_noise is None:
                    self.noise_samples[:,1] =  self.particles[:,-2]
                    self.resample_noise()

                # Compute moments of the weighted samples for resampling
                mean,cov = self.compute_weighted_moments()
                cov += torch.eye(cov.size(0), device=self.particles.device) * 1e-6  # Add a small value to the diagonal for numerical stability

                try:
                    # Resample particles, drawn from a gaussian with computed moments and reset the samples
                    self.particles[:,:-2] = torch.distributions.MultivariateNormal(mean, cov).sample((self.num_particles,))
                    self.particles[:,-2] = 1.0/self.num_particles

                except Exception as e:
                    print(f"Resampling failed with error: {e}")
                    print(f"Gamma: {gamma}")
                    eigvals = torch.linalg.eigvalsh(cov)
                    print("Min Eigenvalue",eigvals.min())  # Should be > 0
                    print("Min diagonal element:", torch.min(torch.diag(cov)))
                    print("Max diagonal element:", torch.max(torch.diag(cov)))
                    raise ValueError("Resampling failed. Check the covariance matrix.")

                # Set likelihood to 0
                self.particles[:,-1] = 0.0

            #Check wether the complete likelihood was used in the end
            assert gamma == 1.0, "The progression loop did not reach the end."

        end = time.time()

        if self.meas_noise is None:
            print("*****Training Finished after ",np.round(end-start,2),f" seconds. Estimated noise: {self.noise_samples[:,0].mean():.2f}+- {self.noise_samples[:,0].std():.2f}*****")
        else:
            print("*****Training Finished after ",np.round(end-start,2)," seconds.*****")

    @torch.no_grad()
    def train_minibatch_progressive(self,x,y,batchsize, prog_step):
        """
        Train the network with the given data !mini-batchwise!.
        Use the Progressive Gauss Filter and random gaussian samples for the resampling.

        """
        print("*****Started Training (Progressive Gauss Filter Random Sampled, Minibatch Method)*****")
        start = time.time()

        # Save original data
        x_orig = x.clone()
        y_orig = y.clone()

        # Compute number of batches
        batchnum = int(x_orig.shape[0]/batchsize)

        # Specify progression parameters
        prog_steps = 1.0/prog_step    #Compute the number of steps based on the manual step size
        gamma_inc = 1.0/prog_steps

        #Iterate through all data in mini batches
        for ii in tqdm(range(batchnum),desc="Mini Batch"):

            # Subsample the complete batch
            x = x_orig[ii*int(x_orig.shape[0]/batchnum):(ii+1)*int(x_orig.shape[0]/batchnum)]
            y = y_orig[ii*int(y_orig.shape[0]/batchnum):(ii+1)*int(y_orig.shape[0]/batchnum)]

            #Progression: Iterate over gamma
            for i in range(1,prog_steps+1):
                # Set temporary gamma (progression exponent)
                gamma = i*gamma_inc

                # Iterate through data points in mini batch
                for pt in range(x.shape[0]):
                    x_seq = x[pt].unsqueeze(0)
                    y_seq = y[pt].unsqueeze(0)

                    #Call Forward Function on all particles (vectorized)
                    _,log_likelihood=self.forward(x_seq,y_seq)

                    # Add Result to log Likelihood
                    self.particles[:, -1] =+log_likelihood
                
                # Shift the log-likeihoods into the positive by adding the minimum value (stability)
                max_val = torch.max(self.particles[:,-1])
                self.particles[:,-1] = self.particles[:,-1] - max_val

                # Exponentiate the log-likelihoods
                self.particles[:,-1] = torch.exp(self.particles[:,-1])    
                
                # Compute the current progression state of the likelihood values
                temp_likelihood = self.particles[:,-1]**gamma_inc
                # Perform the weighting (elementwise)
                self.particles[:,-2] = self.particles[:,-2]*temp_likelihood
                # Normalize the weights
                self.particles[:,-2] = self.particles[:,-2]/torch.sum(self.particles[:,-2])
                
                # Apply the weights to the noise shape as well if it is subject to learning, then resample
                if self.meas_noise is None:
                    self.noise_samples[:,1] =  self.particles[:,-2]
                    self.resample_noise()

                # Compute moments of the weighted samples for resampling
                mean,cov = self.compute_weighted_moments()
                cov += torch.eye(cov.size(0), device=self.particles.device) * 1e-6  # Add a small value to the diagonal for numerical stability
                
                try:
                    # Resample particles, drawn from a gaussian with computed moments and reset the samples
                    self.particles[:,:-2] = torch.distributions.MultivariateNormal(mean, cov).sample((self.num_particles,))
                    self.particles[:,-2] = 1.0/self.num_particles

                except Exception as e:
                    print(f"Resampling failed with error: {e}")
                    print(f"Progressive step: {i}, gamma: {gamma}")
                    eigvals = torch.linalg.eigvalsh(cov)
                    print("Min Eigenvalue",eigvals.min())  # Should be > 0
                    print("Min diagonal element:", torch.min(torch.diag(cov)))
                    print("Max diagonal element:", torch.max(torch.diag(cov)))
                    raise ValueError("Resampling failed. Check the covariance matrix.")
                
                # Set likelihood to 0
                self.particles[:,-1] = 0.0

            #Check wether the complete likelihood was used in the end
            assert gamma == 1.0, "The progression loop did not reach the end."

        end = time.time()

        if self.meas_noise is None:
            print("*****Training Finished after ",np.round(end-start,2),f" seconds. Estimated noise: {self.noise_samples[:,0].mean():.2f}+- {self.noise_samples[:,0].std():.2f}*****")
        else:
            print("*****Training Finished after ",np.round(end-start,2)," seconds.*****")

    @torch.no_grad()
    def train_minibatch_progressive_adaptive(self,x,y,batchsize):
        """
        Train the network with the given data !mini-batchwise!.
        Use the Progressive Gauss Filter and random gaussian samples for the resampling.
        This version uses a automatically determined stepsize based on Steinbring, Hanebeck Publication
        """
        print("*****Started Training (Progressive Gauss Filter Random Sampled, Minibatch Method)*****")
        start = time.time()

        # Save original data
        x_orig = x.clone()
        y_orig = y.clone()

        # Compute number of batches
        batchnum = int(x_orig.shape[0]/batchsize)

        # Specify progression parameters
        log_M = torch.log(torch.tensor(self.num_particles))

        #Iterate through all data in mini batches
        for ii in tqdm(range(batchnum),desc="Mini Batch"):

            # Subsample the complete batch
            x = x_orig[ii*int(x_orig.shape[0]/batchnum):(ii+1)*int(x_orig.shape[0]/batchnum)]
            y = y_orig[ii*int(y_orig.shape[0]/batchnum):(ii+1)*int(y_orig.shape[0]/batchnum)]

            # Set initial gamma
            gamma = 0.0

            #Progression: Iterate over gamma
            while gamma < 1.0:

                # Iterate through data points in mini batch
                for pt in range(x.shape[0]):
                    x_seq = x[pt].unsqueeze(0)
                    y_seq = y[pt].unsqueeze(0)

                    #Call Forward Function on all particles  (vectorized)
                    _,log_likelihood=self.forward(x_seq,y_seq)

                    # Add Result to log Likelihood
                    self.particles[:, -1] =+log_likelihood
                
                # Shift the log-likeihoods into the positive by adding the minimum value (stability)
                max_val = torch.max(self.particles[:,-1])
                self.particles[:,-1] = self.particles[:,-1] - max_val

                #Compute the increment
                min_like = torch.min(self.particles[:,-1])
                max_like = torch.max(self.particles[:,-1])
                
                gamma_inc = -log_M/(min_like-max_like)

                # Ensure gamma_inc is not negative, min gamma_inc is adjustable
                if gamma_inc < 0.0: 
                    gamma_inc = 0.1

                # Update gamma
                if gamma + gamma_inc > 1.0:
                    gamma_inc = 1.0 - gamma
                gamma += gamma_inc

                # Compute the current progression state of the likelihood values
                self.particles[:,-1] = torch.exp(self.particles[:,-1]*gamma_inc)    

                # Perform the weighting (elementwise)
                self.particles[:,-2] = self.particles[:,-2]*self.particles[:,-1]

                # Normalize the weights
                self.particles[:,-2] = self.particles[:,-2]/torch.sum(self.particles[:,-2])
                
                # Apply the weights to the noise shape as well if it is subject to learning, then resample
                if self.meas_noise is None:
                    self.noise_samples[:,1] =  self.particles[:,-2]
                    self.resample_noise()

                # Compute moments of the weighted samples for resampling
                mean,cov = self.compute_weighted_moments()
                cov += torch.eye(cov.size(0), device=self.particles.device) * 1e-6  # Add a small value to the diagonal for numerical stability

                try:
                    # Resample particles, drawn from a gaussian with computed moments and reset the samples
                    self.particles[:,:-2] = torch.distributions.MultivariateNormal(mean, cov).sample((self.num_particles,))
                    self.particles[:,-2] = 1.0/self.num_particles

                except Exception as e:
                    print(f"Resampling failed with error: {e}")
                    print(f"Gamma: {gamma}")
                    eigvals = torch.linalg.eigvalsh(cov)
                    print("Min Eigenvalue",eigvals.min())  # Should be > 0
                    print("Min diagonal element:", torch.min(torch.diag(cov)))
                    print("Max diagonal element:", torch.max(torch.diag(cov)))
                    raise ValueError("Resampling failed. Check the covariance matrix.")
                
                # Set likelihood to 0
                self.particles[:,-1] = 0.0

            #Check wether the complete likelihood was used in the end
            assert gamma == 1.0, "The progression loop did not reach the end."

        end = time.time()

        if self.meas_noise is None:
            print("*****Training Finished after ",np.round(end-start,2),f" seconds. Estimated noise: {self.noise_samples[:,0].mean():.2f}+- {self.noise_samples[:,0].std():.2f}*****")
        else:
            print("*****Training Finished after ",np.round(end-start,2)," seconds.*****")

    @torch.no_grad()
    def train_sequential_progressive_lcd(self,x,y,prog_step):
        """
        Train the network with the given data !sequeniatlly!. Vectorized.
        This Version uses LCD-Samples for the resampling step.
        """
        print("*****Started Training (Progressive Gauss Filter LCD-Sampled, Sequential Method)*****")
        start = time.time()

        # Specify progression parameters
        prog_steps = 1.0/prog_step  #Compute the number of steps based on the manual step size
        gamma_inc = 1.0/prog_steps

        # Iterate through data points
        for pt in tqdm(range(x.shape[0]),desc="Training (Sequential)"):

            # Chose training point and make it (1,d)-Dimensional 
            x_seq = x[pt].unsqueeze(0)
            y_seq = y[pt].unsqueeze(0)

            #Progression: Iterate over gamma
            for i in range(1,prog_steps+1):
                # Set temporary gamma (progression exponent)
                gamma = i*gamma_inc

                #Call Forward Function on all particles  (vectorized)
                _,log_likelihood=self.forward(x_seq,y_seq)

                # Add Result to log Likelihood
                self.particles[:, -1] =+log_likelihood

                # Shift the log-likeihoods into the positive by adding the minimum value (stability)
                max_val = torch.max(self.particles[:,-1])
                self.particles[:,-1] = self.particles[:,-1] - max_val

                # Exponentiate the log-likelihoods
                self.particles[:,-1] = torch.exp(self.particles[:,-1])    

                # Compute the current progression state of the likelihood values
                temp_likelihood = self.particles[:,-1]**gamma_inc

                # Perform the weighting (elementwise)
                self.particles[:,-2] = self.particles[:,-2]*temp_likelihood

                # Normalize the weights
                self.particles[:,-2] = self.particles[:,-2]/torch.sum(self.particles[:,-2])
                
                # Apply the weights to the noise shape as well if it is subject to learning, then resample
                if self.meas_noise is None:
                    self.noise_samples[:,1] =  self.particles[:,-2]
                    self.resample_noise()

                # Compute moments of the weighted samples for resampling
                mean,cov = self.compute_weighted_moments()

                try:
                    # Resample particles, drawn from a gaussian with computed moments and reset the samples
                    self.particles[:,:-2] = self.scale_lcd(mean,cov)
                    self.particles[:,-2] = 1.0/self.num_particles

                except Exception as e:
                    print(f"Resampling failed with error: {e}")
                    print(f"Progressive step: {i}, gamma: {gamma}")
                    eigvals = torch.linalg.eigvalsh(cov)
                    print("Min Eigenvalue",eigvals.min())  # Should be > 0
                    print("Min diagonal element:", torch.min(torch.diag(cov)))
                    print("Max diagonal element:", torch.max(torch.diag(cov)))
                    raise ValueError("Resampling failed. Check the covariance matrix.")

                # Set likelihood to 0
                self.particles[:,-1] = 0.0

            #Check wether the complete likelihood was used in the end
            assert gamma == 1.0, "The progression loop did not reach the end."

        end = time.time()

        if self.meas_noise is None:
            print("*****Training Finished after ",np.round(end-start,2),f" seconds. Estimated noise: {self.noise_samples[:,0].mean():.2f}+- {self.noise_samples[:,0].std():.2f}*****")
        else:
            print("*****Training Finished after ",np.round(end-start,2)," seconds.*****")

    @torch.no_grad()
    def train_minibatch_progressive_lcd(self,x,y,batchsize,prog_step):
        """
        Train the network with the given data !mini-batchwise!. Vectorized.
        This function uses LCD Samples
        """
        print("*****Started Training (Progressive Gauss Filter LCD-Sampled, Minibatch Method)*****")
        start = time.time()

        # Save original data
        x_orig = x.clone()
        y_orig = y.clone()

        # Compute number of batches
        batchnum = int(x_orig.shape[0]/batchsize)

        # Specify progression parameters
        prog_steps = 1.0/prog_step  #Compute the number of steps based on the manual step size
        gamma_inc = 1.0/prog_steps

        #Iterate through all data in mini batches
        for ii in tqdm(range(batchnum),desc="Mini Batch"):

            # Subsample the complete batch
            x = x_orig[ii*int(x_orig.shape[0]/batchnum):(ii+1)*int(x_orig.shape[0]/batchnum)]
            y = y_orig[ii*int(y_orig.shape[0]/batchnum):(ii+1)*int(y_orig.shape[0]/batchnum)]

            #Progression: Iterate over gamma
            for i in range(1,prog_steps+1):
                # Set temporary gamma (progression exponent)
                gamma = i*gamma_inc

                # Iterate through data points in mini batch
                for pt in range(x.shape[0]):
                    x_seq = x[pt].unsqueeze(0)
                    y_seq = y[pt].unsqueeze(0)

                    #Call Forward Function on all particles  (vectorized)
                    _,log_likelihood=self.forward(x_seq,y_seq)

                    # Add Result to log Likelihood
                    self.particles[:, -1] =+log_likelihood
                
                # Shift the log-likeihoods into the positive by adding the minimum value (stability)
                max_val = torch.max(self.particles[:,-1])
                self.particles[:,-1] = self.particles[:,-1] - max_val

                # Exponentiate the log-likelihoods
                self.particles[:,-1] = torch.exp(self.particles[:,-1])    
                
                # Compute the current progression state of the likelihood values
                temp_likelihood = self.particles[:,-1]**gamma_inc
                # Perform the weighting (elementwise)
                self.particles[:,-2] = self.particles[:,-2]*temp_likelihood
                # Normalize the weights
                self.particles[:,-2] = self.particles[:,-2]/torch.sum(self.particles[:,-2])
                
                # Apply the weights to the noise shape as well if it is subject to learning, then resample
                if self.meas_noise is None:
                    self.noise_samples[:,1] =  self.particles[:,-2]
                    self.resample_noise()

                # Compute moments of the weighted samples for resampling
                mean,cov = self.compute_weighted_moments()

                try:
                    # Resample particles, drawn from a gaussian with computed moments and reset the samples
                    self.particles[:,:-2] = self.scale_lcd(mean,cov)
                    self.particles[:,-2] = 1.0/self.num_particles

                except Exception as e:
                    print(f"Resampling failed with error: {e}")
                    print(f"Progressive step: {i}, gamma: {gamma}")
                    eigvals = torch.linalg.eigvalsh(cov)
                    print("Min Eigenvalue",eigvals.min())  # Should be > 0
                    print("Min diagonal element:", torch.min(torch.diag(cov)))
                    print("Max diagonal element:", torch.max(torch.diag(cov)))
                    raise ValueError("Resampling failed. Check the covariance matrix.")
                
                # Set likelihood to 0
                self.particles[:,-1] = 0.0

            #Check wether the complete likelihood was used in the end
            assert gamma == 1.0, "The progression loop did not reach the end."

        end = time.time()

        if self.meas_noise is None:
            print("*****Training Finished after ",np.round(end-start,2),f" seconds. Estimated noise: {self.noise_samples[:,0].mean():.2f}+- {self.noise_samples[:,0].std():.2f}*****")
        else:
            print("*****Training Finished after ",np.round(end-start,2)," seconds.*****")

    @torch.no_grad()
    def train_sequential_progressive_lcd_adaptive(self,x,y):
        """
        Train the network with the given data !sequeniatlly!. Vectorized.
        This Version uses LCD-Samples for the resampling step.
        This version uses a automatically determined stepsize based on Steinbring, Hanebeck Publication
        """
        print("*****Started Training (Progressive Gauss Filter LCD-Sampled, Sequential Method)*****")
        start = time.time()

        # Specify progression parameters
        log_M = torch.log(torch.tensor(self.num_particles))

        # Iterate through data points
        for pt in tqdm(range(x.shape[0]),desc="Training (Sequential)"):

            # Chose training point and make it (1,d)-Dimensional 
            x_seq = x[pt].unsqueeze(0)
            y_seq = y[pt].unsqueeze(0)

            # Set initial gamma
            gamma = 0.0

            #Progression: Iterate over gamma
            while gamma < 1.0:

                #Call Forward Function on all particles  (vectorized)
                _,log_likelihood=self.forward(x_seq,y_seq)

                # Add Result to log Likelihood
                self.particles[:, -1] =+log_likelihood

                # Shift the log-likeihoods into the positive by adding the minimum value (stability)
                max_val = torch.max(self.particles[:,-1])
                self.particles[:,-1] = self.particles[:,-1] - max_val

                #Compute the increment
                min_like = torch.min(self.particles[:,-1])
                max_like = torch.max(self.particles[:,-1])
                
                gamma_inc = -log_M/(min_like-max_like)
                
                # Ensure gamma_inc is not negative, min gamma_inc is adjustable
                if gamma_inc < 0.0: 
                    gamma_inc = 0.1

                # Update gamma
                if gamma + gamma_inc > 1.0:
                    gamma_inc = 1.0 - gamma
                gamma += gamma_inc

                # Compute the current progression state of the likelihood values
                self.particles[:,-1] = torch.exp(self.particles[:,-1]*gamma_inc)

                # Perform the weighting (elementwise)
                self.particles[:,-2] = self.particles[:,-2]*self.particles[:,-1]

                # Normalize the weights
                self.particles[:,-2] = self.particles[:,-2]/torch.sum(self.particles[:,-2])

                # Apply the weights to the noise shape as well if it is subject to learning, then resample
                if self.meas_noise is None:
                    self.noise_samples[:,1] =  self.particles[:,-2]
                    self.resample_noise()

                # Compute moments of the weighted samples for resampling
                mean,cov = self.compute_weighted_moments()

                try:
                    # Resample particles, drawn from a gaussian with computed moments and reset the samples
                    self.particles[:,:-2] = self.scale_lcd(mean,cov)
                    self.particles[:,-2] = 1.0/self.num_particles

                except Exception as e:
                    print(f"Resampling failed with error: {e}")
                    print(f" gamma: {gamma}")
                    eigvals = torch.linalg.eigvalsh(cov)
                    print("Min Eigenvalue",eigvals.min())  # Should be > 0
                    print("Min diagonal element:", torch.min(torch.diag(cov)))
                    print("Max diagonal element:", torch.max(torch.diag(cov)))
                    raise ValueError("Resampling failed. Check the covariance matrix.")

                # Set likelihood to 0
                self.particles[:,-1] = 0.0

            #Check wether the complete likelihood was used in the end
            assert gamma == 1.0, "The progression loop did not reach the end."

        end = time.time()

        if self.meas_noise is None:
            print("*****Training Finished after ",np.round(end-start,2),f" seconds. Estimated noise: {self.noise_samples[:,0].mean():.2f}+- {self.noise_samples[:,0].std():.2f}*****")
        else:
            print("*****Training Finished after ",np.round(end-start,2)," seconds.*****")

    @torch.no_grad()
    def train_minibatch_progressive_lcd_adaptive(self,x,y,batchsize):
        """
        Train the network with the given data !mini-batchwise!. Vectorized.
        This function uses LCD Samples
        This version uses a automatically determined stepsize based on Steinbring, Hanebeck Publication        
        """
        print("*****Started Training (Progressive Gauss Filter LCD-Sampled, Minibatch Method)*****")
        start = time.time()

        # Save original data
        x_orig = x.clone()
        y_orig = y.clone()

        # Compute number of batches
        batchnum = int(x_orig.shape[0]/batchsize)
        
        # Specify progression parameters
        log_M = torch.log(torch.tensor(self.num_particles))

        #Iterate through all data in mini batches
        for ii in tqdm(range(batchnum),desc="Training (Mini Batch)"):

            # Subsample the complete batch
            x = x_orig[ii*int(x_orig.shape[0]/batchnum):(ii+1)*int(x_orig.shape[0]/batchnum)]
            y = y_orig[ii*int(y_orig.shape[0]/batchnum):(ii+1)*int(y_orig.shape[0]/batchnum)]
            
            # Set initial gamma
            gamma = 0.0

            #Progression: Iterate over gamma
            while gamma < 1.0:

                # Iterate through data points in mini batch
                for pt in range(x.shape[0]):
                    x_seq = x[pt].unsqueeze(0)
                    y_seq = y[pt].unsqueeze(0)

                    #Call Forward Function on all particles  (vectorized)
                    _,log_likelihood=self.forward(x_seq,y_seq)

                    # Add Result to log Likelihood
                    self.particles[:, -1] =+log_likelihood
                
                # Shift the log-likeihoods into the positive by adding the minimum value (stability)
                max_val = torch.max(self.particles[:,-1])
                self.particles[:,-1] = self.particles[:,-1] - max_val

                #Compute the increment
                min_like = torch.min(self.particles[:,-1])
                max_like = torch.max(self.particles[:,-1])
                
                gamma_inc = -log_M/(min_like-max_like)

                # Ensure gamma_inc is not negative, min gamma_inc is adjustable
                if gamma_inc < 0.0: 
                    gamma_inc = 0.1

                # Update gamma
                if gamma + gamma_inc > 1.0:
                    gamma_inc = 1.0 - gamma
                gamma += gamma_inc

                # Compute the current progression state of the likelihood values
                self.particles[:,-1] = torch.exp(self.particles[:,-1]*gamma_inc)    

                # Perform the weighting (elementwise)
                self.particles[:,-2] = self.particles[:,-2]*self.particles[:,-1]

                # Normalize the weights
                self.particles[:,-2] = self.particles[:,-2]/torch.sum(self.particles[:,-2])
                
                # Apply the weights to the noise shape as well if it is subject to learning, then resample
                if self.meas_noise is None:
                    self.noise_samples[:,1] =  self.particles[:,-2]
                    self.resample_noise()

                # Compute moments of the weighted samples for resampling
                mean,cov = self.compute_weighted_moments()

                try:
                    # Resample particles, drawn from a gaussian with computed moments and reset the samples
                    self.particles[:,:-2] = self.scale_lcd(mean,cov)
                    self.particles[:,-2] = 1.0/self.num_particles

                except Exception as e:
                    print(f"Resampling failed with error: {e}")
                    print(f"Gamma: {gamma}")
                    eigvals = torch.linalg.eigvalsh(cov)
                    print("Min Eigenvalue",eigvals.min())  # Should be > 0
                    print("Min diagonal element:", torch.min(torch.diag(cov)))
                    print("Max diagonal element:", torch.max(torch.diag(cov)))
                    raise ValueError("Resampling failed. Check the covariance matrix.")
                
                # Set likelihood to 0
                self.particles[:,-1] = 0.0

            #Check wether the complete likelihood was used in the end
            assert gamma == 1.0, "The progression loop did not reach the end."

        end = time.time()

        if self.meas_noise is None:
            print("*****Training Finished after ",np.round(end-start,2),f" seconds. Estimated noise: {self.noise_samples[:,0].mean():.2f}+- {self.noise_samples[:,0].std():.2f}*****")
        else:
            print("*****Training Finished after ",np.round(end-start,2)," seconds.*****")

    @torch.no_grad()
    def forward(self,x,y=None):
        """
        Forward pass, vectorized version
        """
        # First input vector is the input x
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # It is a good question wether i should change that in the input. Incoming, it is always (1,d)-dimensional, but i need it transposed
        z_l = x.T

        # Repeat input vertically for num_particles times, yet without bias (like a vertical stack for vectorized processing)
        z_l = z_l.repeat(self.num_particles, 1)

        # Iterate through the layers (must be done sequentially as they are not independent)
        num_before = 0  #Temporary Variable for tensor indexing
        for l in range(1,self.num_layers):
            # Temporary variables for better oversight
            k = self.num_particles      
            n = self.layers[l-1]+1      
            m = self.layers[l]         
            num_new = n*m               

            # Extract and reshape the weight matrix for the current layer for all particles at the same time.
            # This matrix contains vertical stacked blocks. In each Block, we have (layers[l-1]+1 x layers[l])
            # so combined it is ((num_particles*layers[l-1]+1) x layers[l])
            # The nested Reshape and Permute commands 
            w = self.particles[:,num_before:num_before+num_new].reshape(k, n, m).permute(0, 2, 1).reshape(k * m, n)

            # Temporary Variable for tensor indexing
            num_before += num_new

            # At this point, the z_l Vector should be blocks of (layers[l-1] x 1), so combined it will be ((num_particles*layers[l-1]) x 1)
            # We need to add bias to every block of layers[z_l] so it augments the beforehand layer with a constant bias
            # Add bias row with ones after every self.layers[l-1] rows, so the shape should be ((num_particles*(layers[l-1]+1)) x 1)
            # Reshape so bias can be added as column
            z_l = z_l.view(self.num_particles , self.layers[l-1])
            # Append a bias term (column of ones)
            bias = torch.ones(z_l.size(0), 1, device=z_l.device)
            z_l = torch.cat([z_l, bias], dim=1)
            # Flatten back to a column
            z_l = z_l.view(-1, 1)

            # Perform the forward pass-functions in a vectorized manner
            # Reshape z_l to have a batch dimension for matrix multiplication
            z_l = z_l.view(self.num_particles, self.layers[l-1] + 1, 1)
            # Reshape w to match the batch dimension for matrix multiplication
            w = w.view(self.num_particles, self.layers[l], self.layers[l-1] + 1)
            # Perform batch matrix multiplication, basically every particle with its corresponding input from the before layer
            z_l_next = torch.bmm(w, z_l).view(-1, 1)
            # Apply activation function in a vectorized manner
            z_l = self.act_func[l-1](z_l_next)

        # Compute the likelihood if y is given (Training mode)
        if y is not None:
            # Should the noise be learned?
            if self.meas_noise is None:
                # Treat learnable noise problem
                if not self.layers[-1] == 1:
                    raise ValueError("No learnable noise for multidim. network output as of now")
                else:
                    # Extract the samples
                    noise_samples = self.noise_samples[:,0]

                    # Compute the log likelihood (vectorized, per-particle noise)
                    D = noise_samples.shape[1] if noise_samples.dim() > 1 else 1  # Output dims
                    N = self.num_particles

                    # Reshape z_l to (N, D)
                    z_l = z_l.view(N, D)

                    # Compute deltas
                    delta = y.view(1, D) - z_l  # (N, D)

                    # For each particle, build the covariance matrix and its inverse
                    # If noise_sample is diagonal (no covariance), just use elementwise
                    if D == 1:
                        # Scalar output: noise_sample shape (N, 1) or (N,)
                        inv_noise = 1.0 / noise_samples.view(N, 1)  # (N, 1)
                        maha = (delta ** 2 * inv_noise).sum(dim=1)  # (N,)
                        log_det = torch.log(noise_samples).view(N)   # (N,)
                    else:
                        # Multivariate output: noise_sample shape (N, D)
                        # Assume diagonal covariance for each particle
                        inv_noise = 1.0 / noise_samples  # (N, D)
                        maha = (delta ** 2 * inv_noise).sum(dim=1)  # (N,)
                        log_det = torch.log(noise_samples).sum(dim=1)  # (N,)

                    # Constant term
                    const_term = -0.5 * D * torch.log(2 * self.pi_tensor)

                    # Final log likelihoods (per particle)
                    log_likelihood = const_term - 0.5 * maha - log_det

                    return z_l, log_likelihood
            # Given Noise Covariance Matrix
            else:
                # Compute the log likelihood (vectorized)
                D = self.meas_noise.shape[0]    #Number of Output Dimensions
                N = self.num_particles

                # Reshape z_l to (N, D)
                z_l = z_l.view(N, D)

                # Compute deltas (broadcastring rule backed, y and z don't have same shape)
                delta = y.view(1, D) - z_l

                # Mahalanobis distance for each particle (fast version with indice-swap)
                maha = torch.einsum('nd,dk,nk->n', delta, self.inv_meas_noise, delta)

                # Constant + log det terms
                const_term = -0.5 * D * torch.log(2 * self.pi_tensor)
                log_det = torch.sum(torch.log(torch.diagonal(self.meas_noise)))  # or slogdet for generality

                # Final log likelihoods
                log_likelihood = const_term - 0.5 * maha - log_det
            
                return z_l,log_likelihood

        # Return just net output if no y is given (Prediction mode)
        else:
            # In case noise should be learned
            if self.meas_noise is None:
                if not self.layers[-1] == 1:
                    raise ValueError("No learnable noise for multidim network output as of now")
                
                # Extract constants
                n = self.layers[-1]
                N = self.num_particles

                # Extract per-particle variances (assume shape (N, n))
                noise_samples = self.noise_samples[:,0].view(N, n) 

                # Sample standard normal noise, shape: (N, n)
                noise = torch.randn(N, n, device=z_l.device)

                # Scale by sqrt of per-particle variances to get correct covariance
                noise = noise * torch.sqrt(noise_samples)

                # Reshape z_l to (N, n)
                z_l = z_l.view(N, n)

                # Add noise
                z_l += noise

                # Reshape back to one column
                z_l = z_l.view(N * n, 1)

                # Move to CPU for output, we would not want it on GPU then
                z_l = z_l.cpu()

                return z_l
            
            # Given Noise Covariance Matrix
            else:
                # Extract constants
                n = self.layers[-1]
                N = self.num_particles

                # Sample N noise vectors, shape: (N, n)
                noise = torch.distributions.MultivariateNormal(
                    loc=torch.zeros(n, device=z_l.device),
                    covariance_matrix=self.meas_noise
                ).sample((N,))  

                # Reshape z_l to (N, n)
                z_l = z_l.view(N, n)

                # Add noise
                z_l += noise

                # Reshape back to one column (TACKLE FOR MULTIVAR)
                z_l = z_l.view(N , n)

                # Move to CPU for output, we would not want it on GPU then
                z_l = z_l.cpu()

                return z_l

    @torch.no_grad()
    def filter_step_bootstrap(self):
        """
        Perform the particle filtering for the seen data.
        The likelihood in the particle resembles already the summed up log likelihoods
        !This function is called in vectoized and non vectorized mode!
        """
        # Shift the log-likeihoods into the positive by substracting the maximum value (stability)
        # The maximum value will be negative due to the log likelihood computation, so the result will have its
        # maximum at 0. This is done to avoid numerical issues with the exponentiation.
        temp = self.particles[:,-1]
        max_val = torch.max(self.particles[:,-1])
        self.particles[:,-1] = self.particles[:,-1] - max_val

        # Exponentiate the log-likelihoods
        self.particles[:,-1] = torch.exp(self.particles[:,-1])

        # Weight the weights with the likelihood (elementwise)
        self.particles[:,-2] = self.particles[:,-2]*self.particles[:,-1]

        # Normalize the weights
        self.particles[:,-2] = self.particles[:,-2]/torch.sum(self.particles[:,-2])

        # Set likelihood to 0
        self.particles[:,-1] = 0.0

        # Resample the particles (stratified resampling strategy) (This needs to be checked)
        u = torch.rand(1)/self.num_particles    # Uniform random number. Is this correct?
        u = u.to(self.particles.device)         # For GPU Use
        c = self.particles[:,-2][0]
        i = 0
        new_particles = torch.zeros_like(self.particles)
        for j in range(self.num_particles):
            u_j = u+j/self.num_particles
            while u_j > c:
                i += 1
                if i >= self.num_particles: # Catch over-indexing in some rare cases
                    i = self.num_particles-1
                    break
                c += self.particles[:,-2][i]
            # Numeric noise is added here for the new position
            new_particles[j] = self.particles[i] + torch.randn_like(self.particles[i]) * torch.sqrt(torch.tensor(self.artificial_variance))
            new_particles[j, -2:] = self.particles[i, -2:]  # Preserve the weights and likelihood
            new_particles[j,-2] = 1.0/self.num_particles
        self.particles = new_particles

    @torch.no_grad()
    def predict(self,x):
        """
        Predict the output for the given input.(Vectorized)
        """
        pred = []
        # Move to used device, e.g. CUDA if available and used
        x = x.to(torch.float32)
        x = x.to(self.particles.device)
        
        # Iterate through data points
        for j in tqdm(range(x.shape[0]),desc="Predicting"):
            pred.append(self.forward(x[j].unsqueeze(0)))

        # Stack outputs
        pred = torch.stack(pred).squeeze()
        return pred