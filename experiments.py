"""
Code/methods related to running the actual experiments
"""
import numpy as np
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
from retraining import *
from scipy.stats import sem

def plot_stats(del_sizes, stats, method, err=True, line_style=None):
    """
    Given statistics of interest, computes the mean and generates a
    plot with error bars.

    stats should be supplied with the following convention:
        l = number of different deletion sizes (len(del_sizes))
        r = number of rounds the experiment was run for each group size
        stats[i, j] = value of the statistic for del_sizes[i] on round j
        (Each row contains the data for one deletion group size.)

    Args:
        del_sizes: (list of l ints) List of the number of data points to be deleted
        stats: (l x r matrix) Matrix of statistic values
        method: (string) Method being used for (approximate) deletion
    """
    means = np.mean(stats, axis=1)
    lower_quantiles = np.quantile(stats, 0.25, axis=1)
    upper_quantiles = np.quantile(stats, 0.75, axis=1)
    yerr = sem(stats, axis=1)
    if not err:
        yerr = None
    if line_style == 'dotted':
        plt.errorbar(del_sizes, means, yerr, label=method, linestyle='--')
    else:
        plt.errorbar(del_sizes, means, yerr, label=method)

def gen_plots(del_sizes, param_dists=None, special_weights=None, runtimes=None, saveas=None):
    """
    Generates plots of various statistics of interest vs. group size, with error bars
    param_dists, special_weights, and runtimes all use the same convention:
        i-th row = statistics computed on del_sizes[i]
        l = number of different deletion sizes (len(del_sizes))
        r = number of rounds the experiment was run for each group size
        m = number of methods used (len(methods))

    Args:
        del_sizes: (list of l ints) List of the number of data points to be deleted
        param_dists: (dict of l x r matrices, r = num rounds) Dict of L2 error matrices
        special_weights: (dict of l x r matrices) Dict of special feature weight matrices
        runtimes: (dict of l x r matrices) Dict of runtime matrices
    """
    num_plots = 0
    if param_dists is not None: num_plots += 1
    if special_weights is not None: num_plots += 1
    if runtimes is not None: num_plots += 1

    plt.figure(figsize=(6.0, 4.0)) # set fig size

    current_subplot = 1
    if param_dists is not None:
        param_plot = plt.subplot(1, num_plots, current_subplot)
        param_plot.set_xlabel('Size of removed group')
        param_plot.set_ylabel('Relative $L_2$ parameter distance')
        current_subplot += 1
        for method in param_dists:
            if method == 'no removal':
                line_style='dotted'
            else:
                line_style=None
            plot_stats(del_sizes, param_dists[method], method, line_style=line_style)
        param_plot.legend()
        param_plot.set_xticks(np.arange(10, 110, 10.0))

    if special_weights is not None:
        weight_plot = plt.subplot(1, num_plots, current_subplot)
        weight_plot.set_xlabel('Size of removed group')
        weight_plot.set_ylabel('Relative injected feature weight')
        current_subplot += 1
        for method in special_weights:
            if method == 'no removal':
                line_style='dotted'
            else:
                line_style=None
            if method == 'xs':
                continue
            plot_stats(special_weights['xs'], special_weights[method], method, err=True, line_style=line_style)
        weight_plot.set_xticks(np.arange(20, 120, 20))
        weight_plot.legend()

    if runtimes is not None:
        runtime_plot = plt.subplot(1, num_plots, current_subplot)
        runtime_plot.set_xlabel('Size of removed group')
        runtime_plot.set_ylabel('Runtime')
        current_subplot += 1
        for method in runtimes:
            plot_stats(del_sizes, runtimes[method], method)
        runtime_plot.legend()

    plt.tight_layout()

    if saveas is not None:
        filename = '{}.eps'.format(saveas)
        plt.savefig(filename, bbox_inches='tight', format='eps', dpi=1000)
        print('Saved file as: {}'.format(filename))

    plt.show()
