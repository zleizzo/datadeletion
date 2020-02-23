"""
L2 experiment
"""
import numpy as np
import random
from data.data import *
from retraining import *
from sklearn import datasets
import csv
from scipy.stats import sem

np.random.seed(0)

d = 1500
n = 10 * d
noise = 1
reg = 1e-4

k_vals = [1, 5, 10, 50, 100, 150]
outlier_mult_vals = [1, 3, 10, 30, 100, 300]
R = 100

theta = np.random.randn(d)

param_dists = {
                'base': np.zeros((len(k_vals), len(outlier_mult_vals), R)),
                'res': np.zeros((len(k_vals), len(outlier_mult_vals), R)),
                'inf': np.zeros((len(k_vals), len(outlier_mult_vals), R))
               }


for r in range(R):

    print(f'Running round {r}')
    X = np.random.multivariate_normal(np.zeros(d), datasets.make_spd_matrix(d), n)
    Y = lin_data(X, noise, theta)

    for i in range(len(k_vals)):

        k = k_vals[i]

        for j in range(len(outlier_mult_vals)):

            outlier_multiplier = outlier_mult_vals[j]
            print(f'Deleting group of size {k}, outlier multiplier {outlier_multiplier}')

            X[:k + 1, :] *= outlier_multiplier
            Y[:k + 1] *= outlier_multiplier

            theta_full = lin_exact(X, Y, reg=reg)
            invhess = np.linalg.inv(np.matmul(X.T, X) + reg * np.eye(d))
            H = np.matmul(X, np.matmul(invhess, X.T))

            ind = range(k)
            ind_comp = range(k + 1, n)

            theta_exact = lin_exact(X[ind_comp, :], Y[ind_comp], reg=reg)
            theta_res   = lin_res(X, Y, theta_full, ind, H, reg=reg)
            theta_inf    = lin_inf(X, Y, theta_full, ind, invhess, reg=reg)

            baseline = np.linalg.norm(theta_exact - theta_full)
            param_dists['base'][i, j, r] = baseline
            param_dists['res'][i, j, r]  = np.linalg.norm(theta_exact - theta_res) / baseline
            param_dists['inf'][i, j, r]   = np.linalg.norm(theta_exact - theta_inf) / baseline

            X[:k + 1, :] /= outlier_multiplier
            Y[:k + 1] /= outlier_multiplier


means = {
         'base': np.mean(param_dists['base'], axis=2),
         'res': np.mean(param_dists['res'], axis=2),
         'inf': np.mean(param_dists['inf'], axis=2)
        }

stderr = {
         'res': sem(param_dists['res'], axis=2),
         'inf': sem(param_dists['inf'], axis=2)
         }

Q1 = {
      'res': np.quantile(param_dists['res'], 0.25, axis=2),
      'inf': np.quantile(param_dists['inf'], 0.25, axis=2)
     }

Q2 = {
      'base': np.quantile(param_dists['base'], 0.5, axis=2),
      'res': np.quantile(param_dists['res'], 0.5, axis=2),
      'inf': np.quantile(param_dists['inf'], 0.5, axis=2)
     }

Q3 = {
      'res': np.quantile(param_dists['res'], 0.75, axis=2),
      'inf': np.quantile(param_dists['inf'], 0.75, axis=2)
     }

res = [[None for j in range(len(outlier_mult_vals))] for i in range(len(k_vals))]
inf  = [[None for j in range(len(outlier_mult_vals))] for i in range(len(k_vals))]

res_quantiles = [[None for j in range(len(outlier_mult_vals))] for i in range(len(k_vals))]
inf_quantiles  = [[None for j in range(len(outlier_mult_vals))] for i in range(len(k_vals))]

for i in range(len(k_vals)):

    for j in range(len(outlier_mult_vals)):

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


with open('res_l2.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(res)

with open('inf_l2.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(inf)

with open('res_l2_quant.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(res_quantiles)

with open('inf_l2_quant.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(inf_quantiles)

with open('base_means_l2.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(means['base'])

with open('base_meds_l2.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(Q2['base'])
