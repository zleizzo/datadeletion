from data.process_cifar_data import alt_load_all_cifar_data, load_all_cifar_data
from data.process_yelp_data import load_yelp_data, process_yelp_data
from retraining import *
from experiments import gen_plots
from feature_injection import feature_injection_test
import pickle
import sys
if sys.platform == "darwin": # Apple
   import matplotlib
   matplotlib.use("TkAgg")
   from matplotlib import pyplot as plt
elif 'linux' in sys.platform:
   import matplotlib
   matplotlib.use("agg")
   from matplotlib import pyplot as plt
else:
    from matplotlib import pyplot as plt

def accuracy(theta, X, Y):
    n = len(Y)
    unthresholded_preds = X@theta
    preds = np.zeros(n) - 1
    preds[unthresholded_preds >= 0] = 1
    correct = sum(preds==Y)
    return correct / n

# data = load_yelp_data()
# data = alt_load_all_cifar_data()
data = load_all_cifar_data()
X = data['X']
Y = data['y']
# outliers = data['outliers']
(n, d) = X.shape
repeats = 5
remove_sizes = list(range(5,105,5))
l = len(remove_sizes)
param_dists = {
    'no removal': np.zeros((l, repeats)),
    'residual': np.zeros((l, repeats)),
    'influence': np.zeros((l, repeats)),
}
reg = 1e-4 # pick something reasonable between 0.1 & 0.0001
invhess = np.linalg.inv(np.matmul(X.T, X) + reg * np.eye(d))
H = np.matmul(X, np.matmul(invhess, X.T))

for i in range(len(remove_sizes)):
    k = remove_sizes[i]
    for j in range(repeats):
        indices_to_remove = np.random.choice(range(n), k, replace=False)
        # indices_to_remove = np.random.choice(outliers, k, replace=False)
        indices_to_keep = [i for i in range(n) if i not in indices_to_remove]

        X_retrain = X[indices_to_keep]
        Y_retrain = Y[indices_to_keep]

        # train original classifier:
        theta_original = lin_exact(X, Y, reg=reg)
        # print('Training accuracy of original classifier: {}%'.format( accuracy(theta_original, X, Y)*100 ))

        # retrain from scratch:
        theta_retrain = lin_exact(X_retrain, Y_retrain, reg=reg)
        # print('Training accuracy of retrained classifier: {}%'.format( accuracy(theta_retrain, X_retrain, Y_retrain)*100 ))

        # inf
        theta_inf = lin_inf(X, Y, theta_original, indices_to_remove, invhess, reg=reg)
        # print('Training accuracy of inf classifier: {}%'.format( accuracy(theta_inf, X_retrain, Y_retrain)*100 ))

        # projective residual update
        theta_res = lin_res(X, Y, theta_original, indices_to_remove, H, reg=reg)
        # print('Training accuracy of PRU classifier: {}%'.format( accuracy(theta_res, X_retrain, Y_retrain)*100 ))

        # param distance
        denom = np.linalg.norm(theta_original - theta_retrain)
        param_dists['no removal'][i, j] = 1
        param_dists['influence'][i, j] = np.linalg.norm(theta_inf - theta_retrain) / denom
        param_dists['residual'][i, j] = np.linalg.norm(theta_res - theta_retrain) / denom

# Feature Injection Test
# ws = feature_injection_test(X, Y, remove_sizes=list(range(15,105,5)), num_repeats=repeats, outliers_to_remove=outliers)
ws = feature_injection_test(X, Y, remove_sizes=list(range(5,105,5)), num_repeats=repeats)

save_results_name = 'results.pkl'
results = {
    'remove_sizes': remove_sizes,
    'param_dists': param_dists,
    'special_weights': ws,
}
with open(save_results_name, 'wb') as f:
    pickle.dump(results, f)
print('Results saved to {}.'.format(save_results_name))
with open(save_results_name, 'rb') as f:
    data = pickle.load(f)
SAVE_NAME = 'cifar' # 'yelp'
gen_plots(data['remove_sizes'], param_dists=data['param_dists'], saveas='figures/{}_l2'.format(SAVE_NAME))
gen_plots(data['remove_sizes'], special_weights=data['special_weights'], saveas='figures/{}_w'.format(SAVE_NAME))
