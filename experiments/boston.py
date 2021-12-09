import torch

params_dict = dict(seed=[0],
                   epochs=[20000],
                   genf=["BNN"],
                   bnn_structure=[[10, 10]],
                   regression_coeffs=[20],
                   lr=[0.001],
                   vip_layers=[[1], [1, 1], [4, 1], [7, 1]],
                   activation=["tanh", "cos"],
                   dataset_name=["boston"],
                   verbose=[0],
                   fix_prior_noise=[True],
                   batch_size=[400],
                   dtype=[torch.float64])
