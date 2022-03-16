import os

os.chdir('/home/wenyu/pycharm_projs/Categorical_encoding/')

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
from botorch.acquisition.analytic import ProbabilityOfImprovement
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

from sklearn.preprocessing import PolynomialFeatures


def del_tensor_element(tensor, index):
    t1 = tensor[0:index]
    t2 = tensor[index + 1:]
    return torch.cat((t1, t2), dim=0)


f = test_func.mlp_mse
encoder = 'Esembling'
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

max_trials = 1
budget = 200
init_N = 24
df_list = []
sca_z = preprocessing.MinMaxScaler(feature_range=(-1, 1))
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
init_data_path = './experiment_data/benchmark/init_data/mlp/'

design_obj = initBO(objfn=f, initN=design_seed, bounds=bounds, C=C, saving_path=init_data_path)
design_data = design_obj.sampling_at_least_once_z(seed=design_seed)

init_obj = initBO(f, init_N, bounds, C, saving_path=init_data_path)
data, y = init_obj.initialise(seed=1)
y = -1 * y
y_s = scaler.fit_transform(y)
y_enc = y.numpy()

from sktools import QuantileEncoder, SummaryEncoder
from category_encoders import TargetEncoder


class esembling_encoder():

    def __init__(self, cols=None, quantile=0.8, quantile_tuple=(0.75), moment_tuple=(1)):
        self.cols = cols
        self.quantile = quantile
        self.quantile_tuple = (0.25, 0.75)
        self.moment_tuple = (1)
        self.enc_list = None
        self.X_r = None

    def transform(self, X, y=None):
        return self.X_r

    def fit(self, X, y, randm=False):
        X_encoded = pd.DataFrame(X.copy())
        m_enc = TargetEncoder(cols=self.cols, randomized=True)
        m_enc.fit(X, y)
        X_m = m_enc.transform(X)

        q_enc = QuantileEncoder(cols=self.cols, quantile=self.quantile)
        q_enc.fit(X, y)
        X_q = q_enc.transform(X)
        for col in self.cols:
            X_encoded[col] = 0.5 * (X_m[col] + X_q[col])

        self.X_r = X_encoded
        return self


class multi_stats_encoder():
    def __init__(self, cols=None, quantiles=[0.25, 0.75], randomized=True, sigma=0.01):
        self.cols = cols
        self.quantiles = quantiles
        self.randomized = randomized
        self.sigma = sigma
        # self.mapping = {0: {}, 1: {}}
        self.mapping0 = {}
        self.mapping1 = {}
    def fit(self, X, y):
        # generate mapping
        for col in self.cols:
            stats0 = y.groupby(X[col]).agg(
                [lambda X: np.quantile(X, self.quantiles[0])]).iloc[:, -1]
            stats1 = y.groupby(X[col]).agg(
                [lambda X: np.quantile(X, self.quantiles[1])]).iloc[:, -1]
            self.mapping0[col] = stats0
            self.mapping1[col] = stats1

        return self

    def transform(self, X):
        col_encoded = len(self.cols)*2
        X_encoded = pd.DataFrame(data=np.zeros(shape=(X.shape[0], col_encoded)))
        # for col in range(len(self.cols) * 2):

        # for col in self.cols:
        #     X[col] = X[col].map(self.mapping[col])

        for col in self.cols:
            X_encoded[col*2] = X[col].map(self.mapping0[col])
            X_encoded[col*2+1] = X[col].map(self.mapping1[col])
            # add noise
            if self.randomized:
                X_encoded[col*2] = (X_encoded[col*2] + np.random.normal(0, self.sigma, X_encoded.shape[0]))
                X_encoded[col*2+1] = (X_encoded[col*2+1] + np.random.normal(0, self.sigma, X_encoded.shape[0]))

# if self.randomized:
#     print("--------add noise-----", "   sigma=",self.sigma)
#     for col in self.cols:
#         # X[col] = X[col].map(self.mapping[col])
#         # Randomization is meaningful only for training data -> we do it only if y is present
#         X[col] = (X[col] + np.random.normal(0, self.sigma, X[col].shape[0])


        X_continuous = X.iloc[:, len(self.cols):]
        X_continuous.columns = range(2*len(self.cols)-1,  len(self.cols)+X.shape[1]-1)
        X_encoded = pd.concat([X_encoded, X_continuous], axis=1)


        # for col in range(X_encoded.shape[1] - len(self.cols) * 2):
        # X_encoded[len(self.mapping) * 2 + col] = X[len(self.mapping) + col]

        return X_encoded


# EE = esembling_encoder(cols=list(range(len(C)))).fit(data, y.numpy())
# z = EE.transform(data)

ms = multi_stats_encoder(cols=list(range(len(C)))).fit(X=pd.DataFrame(data), y=pd.Series(y.numpy().reshape(-1)))
z = ms.transform(pd.DataFrame(data))
print("end")
