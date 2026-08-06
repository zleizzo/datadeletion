

# Data Deletion for Linear Regression

Code and experiments for evaluating approximate data deletion (machine unlearning) methods in regularized linear regression. This repository compares exact retraining against several approximation techniques, including Influence Functions, Projective Residual Updates (LKO), and Newton-step updates. It also provides utilities for processing real-world datasets and generating statistical/timing benchmarks.

## Installation
This project requires Python 3.6+ and the following scientific libraries:
```bash
pip install numpy scipy matplotlib scikit-learn torch torchvision
```
No additional configuration is required. The code automatically handles `matplotlib` backend selection for macOS and Linux environments.

## Usage & Experiments
All experiments are designed to be run end-to-end via Python scripts. They generate results, save them to CSVs, and produce publication-ready EPS plots.

### 1. Run Synthetic Experiments
Three primary scripts benchmark different aspects of the deletion methods:
- `python synthetic_l2_final.py`: Evaluates L2 parameter error under varying deletion sizes and outlier multipliers.
- `python synthetic_special_feature_final.py`: Tests feature retention/sparsification effects when deleting groups of points.
- `python runtime_experiment_final.py`: Benchmarks runtime overhead of approximate methods vs. exact retraining across feature dimensions and deletion sizes.

Each script outputs result tables (`.csv`) and automatically generates corresponding figures in the `figures/` directory.

### 2. Process Real-World Data
The `data/` module contains utilities for CIFAR-100 and Yelp datasets:
- **CIFAR-100**: `python data/process_cifar_data.py` downloads, filters, and extracts AlexNet features from the CIFAR-100 dataset.
- **Yelp**: `python data/process_yelp_data.py` tokenizes and vectorizes Yelp reviews into a bag-of-words matrix.
- Processed data is saved as `.pkl` files for use in downstream experiments.

### 3. Real Data Evaluation
Run `python real_data_experiments.py` to evaluate deletion methods on pre-processed CIFAR/Yelp features. It computes parameter distance metrics, runs feature injection tests, and saves results to `results.pkl`.

### 4. Visualization
Use the plotting utilities in `experiments.py`:
```python
from experiments import gen_plots
import pickle

with open('results.pkl', 'rb') as f:
    data = pickle.load(f)

gen_plots(data['remove_sizes'], param_dists=data['param_dists'], saveas='figures/l2_error')
gen_plots(data['remove_sizes'], special_weights=data['special_weights'], saveas='figures/special_weights')
```

## Core Methods
The `retraining.py` module implements the following linear regression deletion/update algorithms:
- `lin_exact`: Exact retraining via normal equations.
- `lin_inf`: Approximate update using Influence Functions (gradient-based first-order approximation).
- `lin_res`: Projective Residual Update (approximates Leave-One-Out/LKO predictions).
- `lin_newton`: Approximate update using a single Newton step with Sherman-Morrison-Woodbury inversion.

## Repository Structure
```
.
├── retraining.py            # Core deletion algorithms (exact, influence, residual, newton)
├── experiments.py           # Plotting and statistics visualization utilities
├── real_data_experiments.py # End-to-end evaluation on real datasets
├── synthetic_*.py           # Synthetic benchmark scripts (L2 error, special features, runtime)
├── data/
│   ├── data.py              # Synthetic linear data generator
│   ├── process_cifar_data.py # CIFAR-100 preprocessing & feature extraction
│   └── process_yelp_data.py  # Yelp review tokenization & vectorization
└── figures/                 # Output directory for generated plots
```

All experiments assume a standard Unix/macOS environment with sufficient RAM for matrix operations. For large-scale runs, adjust `num_repeats` and dimension parameters in the respective scripts.
