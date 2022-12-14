"""Hartmann-6 function"""
import numpy as np
import scipy
import GPyOpt
import GPy
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
from maEI import maEI
from parameter_distribution import ParameterDistribution
from utility import Utility
import cbo
import sys
import time
import os

def get_trial_dir(dir_format, i0=0):
    i = i0
    while True:
        results_dir_i = dir_format % i
        if os.path.isdir(results_dir_i):
            i += 1
        else:
            try:
                os.makedirs(results_dir_i)
                break
            except FileExistsError:
                pass
    return results_dir_i, i

time_list = []
x_list = []
y_list = []

results_dir_i, _ = get_trial_dir('results/hartmann6/bocf/trial%d')

# --- Function to optimize
alpha = [1.0, 1.2, 3.0, 3.2]
A = np.atleast_2d([[10, 3, 17, 3.5, 1.7, 8],
     [0.05, 10, 17, 0.1, 8, 14],
     [3, 3.5, 1.7, 10, 17, 8],
     [17, 8, 0.05, 10, 0.1, 14]])
P = 10**(-4) * np.atleast_2d([[1312, 1696, 5569, 124, 8283, 5886],
                 [2329, 4135, 8307, 3736, 1004, 9991],
                 [2348, 1451, 3522, 2883, 3047, 6650],
                 [4047, 8828, 8732, 5743, 1091, 381]])

m = 4
def hartman(X):
    X = np.atleast_2d(X)
    hX = np.empty((m,X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(m):
            aux = 0
            for k in range(X.shape[1]):
                aux -= A[j, k]*(X[i,k]-P[j, k])**2
            hX[j,i] = np.exp(aux)

    x_list.append(X)
    time_list.append(time.time())
    y_list.append(np.dot(np.transpose(hX), alpha))
    np.savez(os.path.join(results_dir_i, 'info'), time_list=time_list, x_list=x_list, y_list=y_list)
    return hX

# --- Objective
objective = MultiObjective(hartman, as_list=False, output_dim=m)

# --- Space
space = GPyOpt.Design_space(space=[{'name': 'var', 'type': 'continuous', 'domain': (0, 1), 'dimensionality': 6}])

# --- Model (Multi-output GP)
n_attributes = m
model = multi_outputGP(output_dim=n_attributes, exact_feval=[True] * m, fixed_hyps=False)

# --- Initial design
initial_design = GPyOpt.experiment_design.initial_design('random', space, 14)

# --- Parameter distribution
parameter_support = np.atleast_2d(alpha)
parameter_dist = np.ones((1,)) / 1
parameter_distribution = ParameterDistribution(continuous=False, support=parameter_support, prob_dist=parameter_dist)


# --- Utility function
def U_func(parameter, y):
    return np.dot(parameter,y)


def dU_func(parameter, y):
    # print(np.shape(parameter))
    return parameter


U = Utility(func=U_func,dfunc=dU_func,parameter_dist=parameter_distribution,linear=True)

# --- Compute real optimum value
bounds = [(0, 1)] * 6
starting_points = np.random.rand(100, 6)
opt_val = 0
parameter = parameter_support[0,:]

# def func(x):
#     x_copy = np.atleast_2d(x)
#     fx = hartman(x_copy)
#     # print('test begin')
#     # print(parameter)
#     # print(fx)
#     val = U_func(parameter, fx)
#     return -val
#
#
# best_val_found = np.inf
#
# for x0 in starting_points:
#     res = scipy.optimize.fmin_l_bfgs_b(func, x0, approx_grad=True, bounds=bounds)
#     if best_val_found > res[1]:
#         best_val_found = res[1]
#         x_opt = res[0]
# print('optimum')
# print(x_opt)
# print('optimal value')
# print(-best_val_found)

# --- Acquisition optimizer
acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer='lbfgs2', inner_optimizer='lbfgs2', space=space)

# --- Acquisition function
acquisition = maEI(model, space, optimizer=acq_opt, utility=U)
#acquisition = uKG_cf(model, space, optimizer=acq_opt, utility=U, expectation_utility=expectation_U)

# --- Evaluator
evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

# --- Run CBO algorithm

max_iter = 50
for i in range(1):
    filename = os.path.join(results_dir_i,'results.txt')
    bo_model = cbo.CBO(model, space, objective, acquisition, evaluator, initial_design)
    bo_model.run_optimization(max_iter=max_iter, parallel=False, plot=False, results_file=filename)