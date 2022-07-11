# DeepVIP

Implicit processes (IPs) are a generalization of Gaussian processes (GPs). IPs may lack a closed-form expression but are easy to sample from. Examples include, among others, Bayesian neural networks or neural samplers. IPs can be used as priors over functions, resulting in flexible models with well-calibrated prediction uncertainty estimates. Methods based on IPs usually carry out function-space approximate inference, which overcomes some of the difficulties of parameter-space approximate inference. Nevertheless, the approximations employed often limit the expressiveness of the final model, resulting, e.g., in a Gaussian predictive distribution, which can be restrictive. We propose here a multi-layer generalization of IPs called the Deep Variational Implicit process (DVIP). This generalization is similar to that of deep GPs over GPs, but it is more flexible due to the use of IPs as the prior distribution over the latent functions. We describe a scalable variational inference algorithm for training DVIP and show that it outperforms previous IP-based methods and also deep GPs. We support these claims via extensive regression and classification experiments. We also evaluate DVIP on large datasets with up to several million data instances to illustrate its good scalability and performance. 

## Code organization

The code is divided in several folders that contains Python files of different purposes. More precisely,
- src: Contains the code regarding the implemented model DVIP.
    - `dvip.py`: Contains the definition of the model.
    - `generative_functions.py`: Contains the definition of the different Prior models, such as BNN.
    - `layers_init.py`: Contains a function used to generate the VIP layers using different parameters.
    - `layers.py`: Contains the definition of the VIP layer.
    - `likelihood.py`` : Contains the used likelihoods as Python Classes.
    - `noise_samples.py : Contains classes for Gaussian and Uniform sampling.
    - `quadrature.py`: Includes the necesarry functions to estimate likelihoods using quadrature.
    - `utils.py`: Contains general purpose functions, such as the reparameterization trick for Gaussians.
- scripts: Contains Python scripts that perform different experiments.
    - `split.py`: Runs the necessary code to run an experiment with a precise data split. Saving the obtained metrics.
    - `single_experiment.py`: Performs a single experiment, plotting the convergence curve and showing the obtained metrics.
    - `plotting_experiment.py`: Performs an experiment with 1D data and shows the obtained predictive distribution.
    - `missing_gaps.py`: Performs an experiment with 1D data, used for the CO2 experiment.
    - `filename.py`: Contains a function to create a filename given the parameters of the experiment.
- utils: Contains Python functions that support the execution of experiments.
    - `dataset.py`: Contains a class for every dataset, automatizing their usage.
    - `metrics.py`: Contains classes to handle metrics.
    - `process_flags.py`: Contains the necessary functions to use argparse to handle the experiments' parameters as flags of the python call.
    - `pytorch_learning.py`: Contains loops for training, testing and evaluating a Pytorch model.


## Scripts usage.

Using this files, the different performed experiments can be easily replicable. For example, in order to run a specific split on a UCI dataset:
```
python scripts/split.py --dataset boston  --iterations 150000  --split 0 --bnn_layer SimplerBayesLinear --vip_layers 2
```

CO2 experiments can be done using the specific script:
```
python scripts/missing_gaps.py --dataset CO2 --iterations 100000 --vip_layers 2 --show --bnn_layer SimplerBayesLinear
```

MNIST and Rectangles experiments are done using the same split file, but the split flag has no effect:
```
python scripts/split.py --dataset MNIST --iterations 150000  --split 0 --bnn_layer SimplerBayesLinear --vip_layers 1 --no_input_prop --genf_full_output --genf conv     
```
