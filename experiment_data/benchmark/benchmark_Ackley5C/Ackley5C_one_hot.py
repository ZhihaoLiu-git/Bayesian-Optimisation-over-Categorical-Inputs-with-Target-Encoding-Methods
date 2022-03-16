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


# import os
# print(os.getcwd())

def del_tensor_element(tensor, index):
    t1 = tensor[0:index]
    t2 = tensor[index + 1:]
    return torch.cat((t1, t2), dim=0)


# generate 5W design domain data for Ackley5C
f = test_func.Ackley5C
obj_func = 'Ackley5C'
C = [17, 17, 17, 17, 17]
design_seed = 10000
bounds = [
    {'name': 'h1', 'type': 'categorical', 'domain': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)},
    {'name': 'h2', 'type': 'categorical', 'domain': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)},
    {'name': 'h3', 'type': 'categorical', 'domain': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)},
    {'name': 'h4', 'type': 'categorical', 'domain': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)},
    {'name': 'h5', 'type': 'categorical', 'domain': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)},
    {'name': 'x1', 'type': 'continuous', 'domain': (-1, 1)}]

max_trials = 20
budget = 200
init_N = 24
df_list = []
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
init_data_path = './experiment_data/benchmark/init_data/Ackley5C/'
start = time.perf_counter()
for trial in range(max_trials):
    print("num_trial: ", trial)
    design_seed = 10000
    design_obj = initBO(objfn=f, initN=design_seed, bounds=bounds, C=C, saving_path=init_data_path)
    design_data, design_y = design_obj.initialise(seed=design_seed)
    OHE = OneHotEncoder(cols=list(range(len(C)))).fit(design_data)

    init_obj = initBO(f, init_N, bounds, C, saving_path=init_data_path)
    data, y = init_obj.initialise(seed=trial)
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
        # print("beta: ", beta)
        UCB = UpperConfidenceBound(model=gp, beta=beta)
        try:
            acq_value = UCB(design_z.unsqueeze(-2))
        except:
            print(f"{trial}_error_{iteration}_iteration")
            #             with open(f"./experiment_data/func2C_cholesky_error/{obj_func}_double__q_{quantile_flag}_trial_{trial}_iter_{iteration}", "wb") as file:
            #                 pickle.dump(z, file)
            break

        max_index = torch.argmax(acq_value, dim=-1).tolist()
        maxindex_list.append(max_index)
        # candidate
        candidate_h_x = design_data[max_index]
        candidate_y = design_y[max_index].unsqueeze(0)
        data = np.vstack((data, candidate_h_x))
        y = torch.vstack((y, candidate_y))
        y_s = scaler.fit_transform(y)
        z = OHE.transform(data)
        cand_y_list.append(candidate_y.item())
        max_y_list.append(y.max().item())
        design_data = np.delete(design_data, max_index, axis=0)
        design_y = del_tensor_element(design_y, max_index)

    final_max = max(max_y_list)
    df_list.append([trial, final_max, cand_y_list, max_y_list])

end = time.perf_counter()
duration = end - start
# df = pd.DataFrame(df_list, columns=['num_trial', 'final_max', 'iter_cand_y_list', 'iter_current_max'])
saving_path = f'./experiment_data/benchmark/result/Ackley5C/'
file_name = f"{saving_path}{obj_func}_trials_{max_trials}_budget_{budget}_onehot_ucbdecay_duration{round(duration)}"
output = open(file_name, 'wb')
pickle.dump(df_list, output)
output.close()
print("-----------duration=", duration)

