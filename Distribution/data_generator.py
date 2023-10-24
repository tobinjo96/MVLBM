import numpy as np
from sklearn.utils import check_random_state
from numbers import Number
from copy import deepcopy
import sys
sys.path.append('D:/AIM4HEALTH/MVLBM/BOSutils')
import BOSutils.Bos_utils as Bos_utils


def sample_Y(Pi, n_samples, random_state=None):
    
    rng = check_random_state(random_state)
    
    n_views = Pi.ndim
    n_view_components = Pi.shape
    
    y_overall = rng.choice(np.arange(len(Pi.reshape(-1)), dtype=int),
                           size=n_samples,
                           p=Pi.reshape(-1))
    
    Y = np.zeros((n_samples, n_views), dtype=int)
    
    for i in range(n_samples):
        view_idxs = np.unravel_index(indices=y_overall[i],
                                     shape=n_view_components,
                                     order='C')
        for v in range(n_views):
            Y[i, v] = view_idxs[v]
    
    return Y, y_overall


def sample_lbm(view_params, Pi, n_row_samples, n_col_samples, na_pct = 0, random_state=None):
    
    if na_pct >= 1 or na_pct < 0:
        raise ValueError("Invalid Value for na_pct")
    
    n_views = len(view_params)
    
    rng = check_random_state(random_state)
    
    y = {}
    
    # sample cluster labels
    row_Y, row_y_overall = sample_Y(Pi, n_samples=n_row_samples, random_state=rng)
    
    y['row_labels'] = row_y_overall
    
    y['view_labels'] = [{}] * n_views
    
    view_data = []
    
    for v in range(n_views):
        
        n_sets = len(view_params[v]['set parameters'])
        n_set_samples = np.round(n_col_samples[v] * view_params[v]['col_weights']).astype(int)
        n_view_col_samples = np.sum(n_set_samples).astype(int)
        
        cum_set_samples = np.append(0, np.cumsum(n_set_samples))
        X = np.zeros((n_row_samples, n_view_col_samples))
        
        col_Y = []
        y['view_labels'][v]['row_labels'] = row_Y[:, v]
        y['view_labels'][v]['set_labels'] = [{}] * n_sets
        
        for s in range(n_sets):
          col_Y_set = rng.choice(range(3), p = view_params[v]['set parameters'][s]['col_weights'], size = n_set_samples[s])
          y['view_labels'][v]['set_labels'][s]['col_labels'] = col_Y_set
          y['view_labels'][v]['set_labels'][s]['row_labels'] = row_Y[:, v]
          
          X_set = np.zeros((n_row_samples, n_set_samples[s]))
          if 'means' in view_params[v]['set parameters'][s]:
            means = view_params[v]['set parameters'][s]['means']
            covariances = view_params[v]['set parameters'][s]['covariances']
            row_y = row_Y[:, v]
            
            for i in range(n_row_samples):
              for j in range(n_set_samples[s]):
                X_set[i, j] = rng.normal(loc = means[row_y[i], col_Y_set[j]], scale = np.sqrt(covariances[row_y[i], col_Y_set[j]]), size = 1)
          
          if 'modes' in view_params[v]['set parameters'][s]:
            modes = view_params[v]['set parameters'][s]['modes']
            proportions = view_params[v]['set parameters'][s]['proportions']
            row_y = row_Y[:, v]
            pxcpp = Bos_utils.Bos_utils()
            
            for i in range(n_row_samples):
              for j in range(n_set_samples[s]):
                X_set[i, j] = rng.choice(range(3), p = pxcpp.pallx_run(modes[row_y[i], col_Y_set[j]], proportions[row_y[i], col_Y_set[j]], 3), size = 1)
          
          if 'probability' in view_params[v]['set parameters'][s]:
            probability = view_params[v]['set parameters'][s]['probability']
            row_y = row_Y[:, v]
            
            for i in range(n_row_samples):
              for j in range(n_set_samples[s]):
                X_set[i, j] = rng.choice(range(5), p = probability[row_y[i], col_Y_set[j], :], size = 1)
          
          if 'rates' in view_params[v]['set parameters'][s]:
            rates = view_params[v]['set parameters'][s]['rates']
            row_y = row_Y[:, v]
            
            for i in range(n_row_samples):
              for j in range(n_set_samples[s]):
                X_set[i, j] = rng.poisson(rates[row_y[i], col_Y_set[j]], size = 1)
          
          X[:, range(cum_set_samples[s], cum_set_samples[s + 1])] = X_set
          
          col_Y.append(col_Y_set + (3 * s))
          
        na_total = np.round(n_row_samples * n_view_col_samples * na_pct).astype(int)
        row_na_idx = rng.choice(range(n_row_samples), na_total, replace = True)
        col_na_idx = rng.choice(range(n_view_col_samples), na_total, replace = True)
        X[row_na_idx, col_na_idx] = np.nan
        
        view_data.append(X)
        
        y['view_labels'][v]['col_labels'] = np.concatenate(col_Y)
    
    return {'view_data': view_data, 'y': y}
