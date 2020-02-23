from retraining import *

def feature_injection_test(X, Y, remove_sizes, num_repeats=50, reg=1e-4, outliers_to_remove=None):

    (n,d) = X.shape
    ws = {
        'xs':           remove_sizes,
        'no removal':   np.zeros((len(remove_sizes), num_repeats)),
        'residual':     np.zeros((len(remove_sizes), num_repeats)),
        'influence':    np.zeros((len(remove_sizes), num_repeats)),
        # 'retrain':      np.zeros((len(remove_sizes), num_repeats)),
    }
    X_extra_col = np.append(X, np.zeros((n, 1)), axis=1)

    for j in range(num_repeats):
        if outliers_to_remove is None:
            viable_indices = [idx for idx in range(n) if Y[idx] == 1]
        else:
            viable_indices = outliers_to_remove
        special_indices = []
        for i in range(len(ws['xs'])):
            size = ws['xs'][i]
            size_to_select = size - len(special_indices)
            new_selected_indices = np.random.choice(viable_indices, size=size_to_select, replace=False)
            viable_indices = [idx for idx in viable_indices if idx not in new_selected_indices]
            special_indices = special_indices + list(new_selected_indices)

            X_injected = np.copy(X_extra_col)
            X_injected[special_indices, -1] = 1
            indices_to_keep = [idx for idx in range(n) if idx not in special_indices]

            theta_original_extra_col = lin_exact(X_injected, Y, reg=reg)
            theta_retrain_extra_col = lin_exact(X_injected[indices_to_keep], Y[indices_to_keep], reg=reg)
            theta_inf_extra_col = lin_inf(X_injected, Y, theta_original_extra_col, special_indices, reg=reg)
            theta_res_extra_col = lin_res(X_injected, Y, theta_original_extra_col, special_indices, reg=reg)

            ws['no removal'][i, j] = theta_original_extra_col[-1]   / theta_original_extra_col[-1]
            ws['influence'][i, j] = theta_inf_extra_col[-1]          / theta_original_extra_col[-1]
            ws['residual'][i, j] = theta_res_extra_col[-1]          / theta_original_extra_col[-1]
            # ws['retrain'][i, j] = theta_retrain_extra_col[-1]     / theta_original_extra_col[-1]

    return ws
