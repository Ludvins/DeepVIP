# DeepVIP

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
    - `split.py`: Runs the necesarry code to run an experiment with a precise data split. Saving the obtained metrics.
    - `single_experiment.py`: Performs a single experiment, plotting the convergence curve and showing the obtained metrics.
    - `plotting_experiment.py`: Performs an experiment with 1D data and shows the obtained predictive distribution.
    - `extrapolate.py`: Performs an experiment with 1D data, used for the CO2 experiment.
    - `filename.py`: Contains a function to create a filename given the parameters of the experiment.
- utils: Contains Python functions that support the execution of experiments.
    - `dataset.py`: Contains a class for every dataset, automatizing their usage.
    - `metrics.py`: Contains classes to handle metrics.
    - `process_flags.py`: Contains the necesarry functions to use argparse to handle th experiments parameters as flags of the python call.
    - `pytorch_learning.py`: Contains loops for training, testing and evaluating a Pytorch model.


## Scripts usage.

Using this files, the different performed experiments can be easily replicable. For example, in order to run a specific split on a UCI dataset:
```
python scripts/split.py --dataset boston  --iterations 150000  --split 0 --bnn_layer SimplerBayesLinear --vip_layers 2
```

CO2 experiments can be done using the specific script:
```
python scripts\extrapolate.py --dataset CO2 --iterations 100000 --vip_layers 2 --show --bnn_layer SimplerBayesLinear
```
MNIST and Rectangles experiments are done using the same split file, but the split flag has no effect:
```
python scripts/split.py --dataset MNIST --iterations 150000  --split 0 --bnn_layer SimplerBayesLinear --vip_layers 1 --no_input_prop --genf_full_output --genf conv     
```