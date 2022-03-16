# import sys
# sys.path.append("../../")

from initDesignDomain import initBO
from sktools import QuantileEncoder
import syntheticFunctions as test_func
# -------------------START target packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
# -------------------END target packages
import warnings

warnings.filterwarnings("ignore")

# -------------------START Bortorch packages
import torch
import os
import math
import random
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.analytic import ProbabilityOfImprovement
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from sklearn import preprocessing
from category_encoders import OneHotEncoder
plt.rcParams['axes.unicode_minus'] = False
print('Libraries imported')

import pickle
import time

def del_tensor_element(tensor, index):
    t1 = tensor[0:index]
    t2 = tensor[index + 1:]
    return torch.cat((t1, t2), dim=0)

f = test_func.mlp_mse
obj_func = 'mlp'
C = [3, 3, 2, 2]
design_seed = 20000
bounds = [
    {'name': 'activation', 'type': 'categorical', 'domain': tuple(range(3))},
    {'name': 'learning_rate', 'type': 'categorical', 'domain': tuple(range(3))},
    {'name': 'solver', 'type': 'categorical', 'domain': tuple(range(2))},
    {'name': 'early_stopping', 'type': 'categorical', 'domain': tuple(range(3))},
    {'name': 'hidden_layer_sizes', 'type': 'continuous', 'domain': (1, 200)},
    {'name': 'alpha', 'type': 'continuous', 'domain': (0.0001, 1)},
    {'name': 'tol', 'type': 'continuous', 'domain': (0.00001, 1)}]

max_trials = 2
budget = 12
init_N = 24
df_list = []
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
init_data_path = './experiment_data/benchmark/init_data/mlp/'
start = time.perf_counter()
for trial in range(max_trials):
    print("num_trial: ", trial)
    design_obj = initBO(objfn=f, initN=design_seed, bounds=bounds, C=C, saving_path=init_data_path)
    design_data = design_obj.sampling_at_least_once_z(seed=design_seed)
    OHE = OneHotEncoder(cols=list(range(len(C)))).fit(design_data)

    init_obj = initBO(f, init_N, bounds, C, saving_path=init_data_path)
    data, y = init_obj.initialise(seed=trial)
    y = -1 * y
    y_s = scaler.fit_transform(y)
    z = OHE.transform(data)
    maxindex_list = []
    cand_y_list = []
    max_y_list = []
    for iteration in range(budget):
        print("iteration: ", iteration)
        gp = SingleTaskGP(torch.from_numpy(np.array(z)), y).to(dtype=torch.double)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)
        design_z = torch.from_numpy(OHE.transform(design_data).values)
        beta = 20 * 0.985 ** iteration
        UCB = UpperConfidenceBound(model=gp, beta=beta)
        try:
            acq_value = UCB(design_z.unsqueeze(-2))
        except:
            print(f"{trial}_error_{iteration}_iteration")
            break

        max_index = torch.argmax(acq_value, dim=-1).tolist()
        maxindex_list.append(max_index)
        # candidate
        candidate_h_x = design_data[max_index]
        candidate_y = torch.tensor(f(candidate_h_x[:len(C)], candidate_h_x[len(C):])).unsqueeze(0)
        candidate_y = -1 * candidate_y
        data = np.vstack((data, candidate_h_x))
        y = torch.vstack((y, candidate_y))
        y_s = scaler.transform(y)
        z = OHE.transform(data)
        cand_y_list.append(candidate_y.item())
        max_y_list.append(y.max().item())
        design_data = np.delete(design_data, max_index, axis=0)
        # design_y = del_tensor_element(design_y, max_index)

    final_max = max(max_y_list)
    df_list.append([trial, final_max, cand_y_list, max_y_list])
#
# end = time.perf_counter()
# duration = end - start
# saving_path = f'./experiment_data/benchmark/result/mlp/'
# file_name = f"{saving_path}{obj_func}_trials_{max_trials}_budget_{budget}_onehot_ucbdecay_duration{round(duration)}"
# output = open(file_name, 'wb')
# pickle.dump(df_list, output)
# output.close()
# print("-----------duration=", duration)

