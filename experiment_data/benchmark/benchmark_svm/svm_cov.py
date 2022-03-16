import os
# os.chdir ('/home/wenyu/pycharm_projs/Categorical_encoding/')
# os.chdir ('/home/wenyu/pycharm_projs/Categorical_encoding/')
from initDesignDomain import initBO
from sktools import QuantileEncoder, SummaryEncoder
import syntheticFunctions as test_func
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")
# -------------------START Bortorch packages
import torch
import os
import math
import random
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from sklearn import preprocessing
# from category_encoders import TargetEncoder as TE
from category_encoders import TargetEncoder

plt.rcParams['figure.dpi'] = 200
plt.rcParams['axes.unicode_minus'] = False
print('Libraries imported')
# -------------------END Bortorch packages

# -------------------START
import pickle
import time
import copy

def del_tensor_element(tensor, index):
    t1 = tensor[0:index]
    t2 = tensor[index + 1:]
    return torch.cat((t1, t2), dim=0)

encoder = 'mean'
f = test_func.svm_mse
obj_func = 'svm'
C = [3]
design_seed = 10000
bounds = [
    {'name': 'kernel', 'type': 'categorical', 'domain': tuple(range(3))},
    {'name': 'C', 'type': 'continuous', 'domain': (1, 50)},  # C
    {'name': 'epsilon', 'type': 'continuous', 'domain': (0, 1)}]  # epsilon
max_trials = 2
budget = 200
init_N = 24
df_list = []
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
scaler2 = preprocessing.MinMaxScaler(feature_range=(-1, 1))
init_data_path = 'experiment_data/benchmark/init_data/svm/'

start = time.perf_counter()
for trial in range(max_trials):
    print("----------num_trial: ", trial)
    design_obj = initBO(objfn=f, initN=design_seed, bounds=bounds, C=C, saving_path=init_data_path)
    design_data = design_obj.sampling_at_least_once_z(seed=design_seed)

    init_obj = initBO(f, init_N, bounds, C, saving_path=init_data_path)
    data, y = init_obj.initialise(seed=trial)
    y_s = scaler.fit_transform(y)
    # data[:, len(C):] = scaler2.fit_transform(data[:, len(C):])

    TE = TargetEncoder(cols=list(range(len(C)))).fit(data, y_s)
    z = TE.transform(data)

    maxindex_list = []
    cand_y_list = []
    max_y_list = []
    for iteration in range(budget):
        print("iteration: ", iteration)
        design_z = torch.from_numpy(TE.transform(design_data).values)
        # covar_m = RBFKernel(has_lengthscale=False)
        # covar_m = RBFKernel(has_lengthscale=True)
        # gp = SingleTaskGP(covar_module=covar_m, train_X=torch.from_numpy(np.array(z)), train_Y=y).to(dtype=torch.double)
        gp = SingleTaskGP(torch.from_numpy(np.array(z)), y).to(dtype=torch.double)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)
        beta = 20 * 0.985 ** iteration
        UCB = UpperConfidenceBound(model=gp, beta=beta)
        try:
            acq_value = UCB(design_z.unsqueeze(-2))
        except:
            print(f"{trial}_error!!!!!!!!!!!!_{iteration}_iteration")
            break

        max_index = torch.argmax(acq_value, dim=-1).tolist()
        maxindex_list.append(max_index)
        # candidate
        candidate_h_x = design_data[max_index]
        candidate_y = torch.tensor(f([candidate_h_x[0]], candidate_h_x[1:])).unsqueeze(0)
        data = np.vstack((data, candidate_h_x))
        y = torch.vstack((y, candidate_y))
        y_s = scaler.transform(y)
        # data[:, len(C):] = scaler2.transform(data[:, len(C):])
        cand_y_list.append(candidate_y.item())
        max_y_list.append(y.max().item())
        design_data = np.delete(design_data, max_index, axis=0)

        # if iteration > 100:
        z = TE.transform(data)
        cov = gp.covar_module(torch.from_numpy(z.values)).numpy()
        row, col = np.diag_indices_from(cov)
        cov[row, col] = np.zeros([cov.shape[0]])
        # Second order inverse distance weighting
        d_power = cov.sum(axis=1)**(-2)
        d_sum = d_power.sum()
        weight = d_power/d_sum
        print(f"weight  {iteration}", weight)
        y_s = (y_s.reshape(-1) * weight).reshape(-1, 1)


        TE = TargetEncoder(cols=list(range(len(C)))).fit(data, y_s)
        z = TE.transform(data)


    final_max = max(max_y_list)
    df_list.append([trial, final_max, cand_y_list, max_y_list])

