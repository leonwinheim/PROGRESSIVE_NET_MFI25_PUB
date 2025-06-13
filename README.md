# PROGRESSIVE NET (MFI25) PUBLICATION REPO

<p align="left">
    <img src="resources/progressive_net.png" width="100" style="vertical-align:middle; margin-right:10px;">
    <span style="vertical-align:middle; font-size:1.2em;">
        <b>PROGRESSIVE NET</b> &mdash; A BNN trained with a Progressive Gaussian Filter
    </span>
</p>


Welcome to the supplementary repository for the publication 
**"BNN Training as State Estimation: A Progressive Filtering Approach"** 
at MFI 2025, College Station, Texas! The authors Leon Winheim and Uwe D. Hanebeck are with the Intelligent Sensor-Actuator Systems Lab (ISAS) at KIT, Germany.
This repository contains all the code required to reproduce the publicated results for the proposed method. Additionally, you can find a [minimum working example](AA_minimum_working.ipynb) in form of a Jupyter Notebook to get started. 
As the authors of the KBNN paper didn't publish their immplementation yet, the model code is not included here.
If any questions remain unanswered, feel free to contact me under leon.winheim@kit.edu.

## Getting started
You can chose between using the provided Dockerfile to build a container and execute the code there, or just install the dependencies in the requirements.txt file manually and run it without a special environment.
Note the following:

- To reperform the UCI Regression experiments, please first generate csv files of the training data (description in the BB_benchmark_UCI.py file)
- To use LCD samples, read the following subsection

## Generating LCD samples
To use the functionality of LCD based deterministic samples, you first need to generate them. A function for that is not yet available in Python. The two options are:
- Use the Matlab Implementation of Daniel Frisch available on Code-Ocean (https://codeocean.com/capsule/1886845/tree/v1)
- Use the provided MATLAB file "ZZ_DetSampLCD.m" in combination with the nonlinear state estimation Toolbox (https://nonlinearestimation.bitbucket.io/)
In the Matlab-File, the number of dimensions and the sample count need to be specified. The number of dimensions will depend on your network architecture. When you specify a LCD based training procedure, the network will tell you how many dimensions it has and the combination of requested LCD samples as a console log when it cannot find them. Just use that info. After generation, put the csv-file with the particle info into a directory called "big_particles" or set another LCD path manually like in the minimum-working example. The network should be able to find the sample information then.

In both cases, make sure to generate a SND sample set and set the mean and covariance parameters accordingly.

## Current Limitations
- Supported Activation functions are ReLU, Sigmoid and linear (more to come)
- Noise can only be learned in the univariate case

**We plan on appending the functionality of the method and thereafter publishing the code as a Python package.**