"""
Special feature experiment
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
w = 10 # true special weight value

k_vals = [1, 5, 10, 50, 100, 150]
p_vals = [.5, .25, .1, .05] # sparsification parameter; keep a feature nonzero w.p. p; lower p --> more sparse

R = 100

theta = np.random.randn(d)
theta[-1] = 0

special_weight = {
                  'exact': np.zeros((len(k_vals), len(p_vals), R)),
                  'base': np.zeros((len(k_vals), len(p_vals), R)),
                  'res': np.zeros((len(k_vals), len(p_vals), R)),
                  'inf': np.zeros((len(k_vals), len(p_vals), R))
                 }

for i in range(len(k_vals)):

    k = k_vals[i]

    for j in range(len(p_vals)):

        p = p_vals[j]
        print(f'Deleting group of size {k}, sparsification {p}')

        for r in range(R):

            print(f'Running round {r}')

            X = np.random.multivariate_normal(np.zeros(d), datasets.make_spd_matrix(d), n)
            X[k:, -1] = 0
            for b in range(d - 1):
                if random.random() < 1 - p:
                    X[:k, b] = 0

            for a in range(k, n):
                for b in range(d - 1):
                    if random.random() < 1 - p:
                        X[a, b] = 0

            Y = lin_data(X, noise, theta)
            Y[:k] = w * X[:k, -1]

            theta_full = lin_exact(X, Y, reg=reg)
            invhess = np.linalg.inv(np.matmul(X.T, X) + reg * np.eye(d))
            H = np.matmul(X, np.matmul(invhess, X.T))

            ind = range(k)
            ind_comp = range(k + 1, n)

            theta_exact = lin_exact(X[ind_comp, :], Y[ind_comp], reg=reg)
            theta_res   = lin_res(X, Y, theta_full, ind, H, reg=reg)
            theta_inf    = lin_inf(X, Y, theta_full, ind, invhess, reg=reg)

            special_weight['exact'][i, j, r] = theta_exact[-1]
            special_weight['base'][i, j, r]  = theta_full[-1]
            special_weight['res'][i, j, r]   = theta_res[-1] / theta_full[-1]
            special_weight['inf'][i, j, r]    = theta_inf[-1] / theta_full[-1]


means = {
         'exact': np.mean(special_weight['exact'], axis=2),
         'base': np.mean(special_weight['base'], axis=2),
         'res': np.mean(special_weight['res'], axis=2),
         'inf': np.mean(special_weight['inf'], axis=2)
        }

stderr = {
         'res': sem(special_weight['res'], axis=2),
         'inf': sem(special_weight['inf'], axis=2)
        }

Q1 = {
      'res': np.quantile(special_weight['res'], 0.25, axis=2),
      'inf': np.quantile(special_weight['inf'], 0.25, axis=2)
     }

Q2 = {
      'base': np.quantile(special_weight['base'], 0.5, axis=2),
      'res': np.quantile(special_weight['res'], 0.5, axis=2),
      'inf': np.quantile(special_weight['inf'], 0.5, axis=2)
     }

Q3 = {
      'res': np.quantile(special_weight['res'], 0.75, axis=2),
      'inf': np.quantile(special_weight['inf'], 0.75, axis=2)
     }

res = [[None for j in range(len(p_vals))] for i in range(len(k_vals))]
inf  = [[None for j in range(len(p_vals))] for i in range(len(k_vals))]

res_quantiles = [[None for j in range(len(p_vals))] for i in range(len(k_vals))]
inf_quantiles  = [[None for j in range(len(p_vals))] for i in range(len(k_vals))]


for i in range(len(k_vals)):

    for j in range(len(p_vals)):

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


with open('res_special_weight.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(res)

with open('inf_special_weight.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(inf)

with open('res_special_weight_quant.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(res_quantiles)

with open('inf_special_weight_quant.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(inf_quantiles)

with open('base_means_special_weight.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(means['base'])

with open('base_meds_special_weight.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(Q2['base'])

with open('exact_special_weights.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(means['exact'])
