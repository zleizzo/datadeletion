"""
Timing experiment
"""
import numpy as np
import random
from data.data import *
from retraining import *
from timeit import timeit
import csv
from scipy.stats import sem

np.random.seed(0)

d_vals = [1000, 1500, 2000, 2500, 3000]
k_vals = [1, 2, 3, 4, 5, 10, 25, 50, 75, 100, 125, 150]

noise = 1
R = 100
timeit_number = 1

runtimes = {
            'exact': np.zeros((len(d_vals), len(k_vals), R)),
            'res': np.zeros((len(d_vals), len(k_vals), R)),
            'inf': np.zeros((len(d_vals), len(k_vals), R))
            }

for i in range(len(d_vals)):

    d = d_vals[i]
    n = 10 * d

    theta = np.random.randn(d)
    X = np.random.multivariate_normal(np.zeros(d), datasets.make_spd_matrix(d), n)
    Y = lin_data(X, noise, theta)

    theta_full = lin_exact(X, Y)
    invhess = np.linalg.inv(np.matmul(X.T, X))
    H = np.matmul(X, np.matmul(invhess, X.T))

    for j in range(len(k_vals)):

        k = k_vals[j]

        for r in range(R):

            ind = range(k)
            ind_comp = range(k + 1, n)

            def test():
                return lin_newton(X, Y, theta_full, ind, invhess)
            T = timeit("test()", setup="from __main__ import test", number=timeit_number) / timeit_number
            runtimes['exact'][i, j, r] = T

            def test():
                return lin_res(X, Y, theta_full, ind, H)
            t = timeit("test()", setup="from __main__ import test", number=timeit_number) / timeit_number
            runtimes['res'][i, j, r] = t / T

            def test():
                return lin_inf(X, Y, theta_full, ind, invhess)
            t = timeit("test()", setup="from __main__ import test", number=timeit_number) / timeit_number
            runtimes['inf'][i, j, r] = t / T


means = {
         'exact': np.mean(runtimes['exact'], axis=2),
         'res': np.mean(runtimes['res'], axis=2),
         'inf': np.mean(runtimes['inf'], axis=2)
        }

stderr = {
         'res': sem(runtimes['res'], axis=2),
         'inf': sem(runtimes['inf'], axis=2)
         }

Q1 = {
      'res': np.quantile(runtimes['res'], 0.25, axis=2),
      'inf': np.quantile(runtimes['inf'], 0.25, axis=2)
     }

Q2 = {
      'exact': np.quantile(runtimes['exact'], 0.5, axis=2),
      'res': np.quantile(runtimes['res'], 0.5, axis=2),
      'inf': np.quantile(runtimes['inf'], 0.5, axis=2)
     }

Q3 = {
      'res': np.quantile(runtimes['res'], 0.75, axis=2),
      'inf': np.quantile(runtimes['inf'], 0.75, axis=2)
     }

res = [[None for j in range(len(k_vals))] for i in range(len(d_vals))]
inf  = [[None for j in range(len(k_vals))] for i in range(len(d_vals))]

res_quantiles = [[None for j in range(len(k_vals))] for i in range(len(d_vals))]
inf_quantiles  = [[None for j in range(len(k_vals))] for i in range(len(d_vals))]


for i in range(len(d_vals)):

    for j in range(len(k_vals)):

        res_mean = round(means['res'][i, j], 6)
        inf_mean  = round(means['inf'][i, j], 6)

        res_dev = round(stderr['res'][i, j], 6)
        inf_dev  = round(stderr['inf'][i, j], 6)

        res[i][j] = f'{res_mean} \pm {res_dev}'
        inf[i][j]  = f'{inf_mean} \pm {inf_dev}'

        res_Q1 = round(Q1['res'][i, j], 6)
        res_Q2 = round(Q2['res'][i, j], 6)
        res_Q3 = round(Q3['res'][i, j], 6)

        inf_Q1  = round(Q1['inf'][i, j], 6)
        inf_Q2  = round(Q2['inf'][i, j], 6)
        inf_Q3  = round(Q3['inf'][i, j], 6)

        res_quantiles[i][j] = f'{res_Q2} ({res_Q1} - {res_Q3})'
        inf_quantiles[i][j]  = f'{inf_Q2} ({inf_Q1} - {inf_Q3})'


with open('res_runtime.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(res)

with open('inf_runtime.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(inf)

with open('res_runtime_quant.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(res_quantiles)

with open('inf_runtime_quant.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(inf_quantiles)

with open('exact_means_runtime.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(means['exact'])

with open('exact_meds_runtime.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(Q2['exact'])
