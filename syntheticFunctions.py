# -*- coding: utf-8 -*-
# ==========================================
# Title:  syntheticFunctions.py
# Author: Binxin Ru and Ahsan Alvi
# Date:   20 August 2019
# Link:   https://arxiv.org/abs/1906.08878
# ==========================================

import numpy as np


# =============================================================================
# Rosenbrock Function (f_min = 0)
# https://www.sfu.ca/~ssurjano/rosen.html
# =============================================================================
def myrosenbrock(X):
    X = np.asarray(X)
    X = X.reshape((-1, 2))
    if len(X.shape) == 1:  # one observation
        x1 = X[0]
        x2 = X[1]
    else:  # multiple observations
        x1 = X[:, 0]
        x2 = X[:, 1]
    fx = 100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2
    return fx.reshape(-1, 1) / 300


# =============================================================================
#  Six-hump Camel Function (f_min = - 1.0316 )
#  https://www.sfu.ca/~ssurjano/camel6.html       
# =============================================================================
def mysixhumpcamp(X):
    X = np.asarray(X)
    X = np.reshape(X, (-1, 2))
    if len(X.shape) == 1:
        x1 = X[0]
        x2 = X[1]
    else:
        x1 = X[:, 0]
        x2 = X[:, 1]
    term1 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2 ** 2) * x2 ** 2
    fval = term1 + term2 + term3
    return fval.reshape(-1, 1) / 10


# =============================================================================
# Beale function (f_min = 0)
# https://www.sfu.ca/~ssurjano/beale.html
# =============================================================================
def mybeale(X):
    X = np.asarray(X) / 2
    X = X.reshape((-1, 2))
    if len(X.shape) == 1:
        x1 = X[0] * 2
        x2 = X[1] * 2
    else:
        x1 = X[:, 0] * 2
        x2 = X[:, 1] * 2
    fval = (1.5 - x1 + x1 * x2) ** 2 + (2.25 - x1 + x1 * x2 ** 2) ** 2 + (
            2.625 - x1 + x1 * x2 ** 3) ** 2
    return fval.reshape(-1, 1) / 50


def func2C(ht_list, X):
    # ht is a categorical index
    # X is a continuous variable
    X = X * 2

    assert len(ht_list) == 2
    ht1 = ht_list[0]
    ht2 = ht_list[1]

    f = 0
    if ht1 == 0:  # rosenbrock
        f = myrosenbrock(X)
    elif ht1 == 1:  # six hump
        f = mysixhumpcamp(X)
    elif ht1 == 2:  # beale
        f = mybeale(X)

    if ht2 == 0:  # rosenbrock
        f = f + myrosenbrock(X)
    elif ht2 == 1:  # six hump
        f = f + mysixhumpcamp(X)
    else:
        f = f + mybeale(X)
    # print("f.shape is ", f.shape) [2,1]
    y = f + 1e-6 * np.random.rand(f.shape[0], f.shape[1])
    # y = -(f + 1e-6 * np.random.rand(f.shape[0], f.shape[1]))
    return y.astype(float)


def func3C(ht_list, X):
    # ht is a categorical index
    # X is a continuous variable
    X = np.atleast_2d(X)
    assert len(ht_list) == 3
    ht1 = ht_list[0]
    ht2 = ht_list[1]
    ht3 = ht_list[2]

    X = X * 2
    if ht1 == 0:  # rosenbrock
        f = myrosenbrock(X)
    elif ht1 == 1:  # six hump
        f = mysixhumpcamp(X)
    elif ht1 == 2:  # beale
        f = mybeale(X)

    if ht2 == 0:  # rosenbrock
        f = f + myrosenbrock(X)
    elif ht2 == 1:  # six hump
        f = f + mysixhumpcamp(X)
    else:
        f = f + mybeale(X)

    if ht3 == 0:  # rosenbrock
        f = f + 5 * mysixhumpcamp(X)
    elif ht3 == 1:  # six hump
        f = f + 2 * myrosenbrock(X)
    else:
        f = f + ht3 * mybeale(X)

    y = f + 1e-6 * np.random.rand(f.shape[0], f.shape[1])

    return y.astype(float)


def Ackley5C(h_list, d):
# ndarry   h_arr:[1,2,...,17],  d: 1-dimension
    h_arr = np.array(h_list)
    x_cate = -1 + 1/8*(h_arr - 1)
    x = np.hstack((x_cate, d))
    a = 20
    b= 0.2
    c = 2*np.pi
    dim = x.shape[0]
    sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(np.square(x))/dim))
    cos_term = -1*np.exp(np.sum(np.cos(c*np.copy(x))/dim))
    result = sum_sq_term + cos_term + a + np.exp(1)
    return result + 1e-6 * np.random.rand()

def Ackley9C(h_list, d):
    h_arr = np.array(h_list)
    x_cate = -1 + 0.025*(h_arr)
    x = np.hstack((x_cate, d))
    a = 20
    b= 0.2
    c = 2*np.pi
    dim = x.shape[0]
    sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(np.square(x))/dim))
    cos_term = -1*np.exp(np.sum(np.cos(c*np.copy(x))/dim))
    result = sum_sq_term + cos_term + a + np.exp(1)
    return result + 1e-6 * np.random.rand()

def twogaussian(shifter=0, x=0):
    # for the function, domain of definition should be:x [-2, 10], C [25, 50, 100]
    # as is transformed to [-1, 1], x should be reverse normalize
    shifter = shifter[0]
    y = np.exp(-(x-(shifter*0.1)-2)**2)+np.exp(-(x-(shifter*0.1)-6)**2/10)+1/((x+(shifter*0.1))**2+1)
    return y.reshape(-1, 1) + shifter # Min=0.2, Max=1.4

def svm_mse(h_list, x):
    from sklearn.model_selection import cross_val_score
    from sklearn.svm import SVR
    from sklearn import datasets
    X, y = datasets.load_boston(return_X_y=True)
    if h_list[0] == 0:
        k = 'poly'
    elif h_list[0] == 1:
        k = 'rbf'
    else:
        k = 'sigmoid'
    c = x[0]
    epsilon = x[1]
    reg = SVR(gamma='scale', C=c, kernel=k, epsilon=epsilon)
    # BO is searching the maximum, delete the minus of score
    score = np.mean(cross_val_score(reg, X, y, cv=3, n_jobs=4,
                                    scoring="neg_mean_squared_error"))
    log_score = np.log(np.absolute(score))
    print("log_score:", log_score)
    return log_score

def mlp_mse(h_list, x):
    from sklearn.model_selection import cross_val_score
    from sklearn.neural_network import MLPRegressor
    from sklearn import datasets
    # print("h_list:", h_list, "x", x)
    X, y = datasets.load_boston(return_X_y=True)
    switch_act_fun = {0: 'logistic', 1: 'tanh', 2: 'relu'}
    switch_learning = {0: 'constant', 1: 'invscaling', 2: 'adaptive'}
    switch_solver = {0: 'sgd', 1: 'adam'}
    switch_stopping = {0: True, 1: False}
    act = switch_act_fun[h_list[0]]
    lea = switch_learning[h_list[1]]
    sol = switch_solver[h_list[2]]
    sto = switch_stopping[h_list[3]]
    print("h_list", h_list, "x", x)
    mlp_reg = MLPRegressor(random_state=0, activation=act, learning_rate=lea, solver=sol, early_stopping=sto,
                           hidden_layer_sizes=int(x[0]), alpha=x[1], tol=x[2], max_iter=2000)
    # print("mpl_reg:", mlp_reg
    # BO is searching the maximum, delete the minus of score
    score = np.mean(cross_val_score(mlp_reg, X, y, cv=3, n_jobs=4,
                                    scoring="neg_mean_squared_error"))
    # return score
    log_score = np.log(np.absolute(score))
    print("score:", score)
    print("log_score:", log_score)
    return log_score

def Hartmann6(h_list, x):
    """Hartmann6 function (6-dimensional with 1 global minimum and 6 local minimum)
    minimums = [(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)]
    fmin = -3.32237,   fmax = 0.0,
    4Categorical  + 2 continuous, h = [1,2, ..., 17], x (0, 1)
    """
    h_arr = np.array(h_list)
    x_cate = -1 + 1 / 8 * (h_arr - 1)
    x = np.hstack((x_cate, x))

    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
    )
    P = 10 ** (-4) * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )
    y = 0.0
    for j, alpha_j in enumerate(alpha):
        t = 0
        for k in range(6):
            t += A[j, k] * ((x[k] - P[j, k]) ** 2)
        y -= alpha_j * np.exp(-t)
    # return result + 1e-6 * np.random.rand()
    return y

def michalewicz(h_list , x):  # mich.m
    " 7Categorical  + 3 continuous, h = [1,2, ..., 16], x (0, 3) "
    from numpy import pi, sin
    h_arr = np.array(h_list)
    x_cate = 1 / 5 * (h_arr - 1)
    x = np.hstack((x_cate, x))

    michalewicz_m = 10  # orig 10: ^20 => underflow
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange(1., n + 1)
    return - sum(sin(x) * sin(j * x ** 2 / pi) ** (2 * michalewicz_m))