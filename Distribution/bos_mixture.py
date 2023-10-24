"""
This module is a lightly modifed version of sklearn.mixture.GaussianMixture().
"""

import numpy as np
#from Bos_utils import *
from scipy import linalg
from scipy.stats import mode
from scipy.special import factorial
from sklearn.mixture._base import _check_shape
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import row_norms
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, DensityMixin
from warnings import warn
from textwrap import dedent

from base import EMfitMMMixin, MixtureModelMixin, _em_docs

import sys
sys.path.append('D:/AIM4HEALTH/MVLBM/BOSutils')
import BOSutils.Bos_utils as Bos_utils

# import rpy2.robjects as robjects
# from rpy2.robjects.packages import STAP
# 
# with open('D:/AIM4HEALTH/MVLBM/pej.R', 'r') as f:
#     string = f.read()
# 
# pej_R = STAP(string, "pej")
# 
# 

###############################################################################
# Latent Block Model shape checkers used by the LatentBlockModel class

def _check_weights(weights, n_components):
    """Check the user provided 'weights'.

    Parameters
    ----------
    weights : array-like, shape (n_components,)
        The proportions of components of each mixture.

    n_components : int
        Number of components.

    Returns
    -------
    weights : array, shape (n_components,)
    """
    weights = check_array(weights, dtype=[np.float64, np.float32],
                          ensure_2d=False)
    _check_shape(weights, (n_components,), 'weights')

    # check range
    if (any(np.less(weights, 0.)) or
            any(np.greater(weights, 1.))):
        raise ValueError("The parameter 'weights' should be in the range "
                         "[0, 1], but got min value %.5f, max value %.5f"
                         % (np.min(weights), np.max(weights)))

    # check normalization
    if not np.allclose(np.abs(1. - np.sum(weights)), 0.):
        raise ValueError("The parameter 'weights' should be normalized, "
                         "but got sum(weights) = %.5f" % np.sum(weights))
    return weights

def _check_modes(modes, n_row_components, n_col_components):
    """Validate the provided 'means'.

    Parameters
    ----------
    probabilty : array-like, shape (n_row_components, n_col_components)
        The centers of the current components.

    n_row_components : int
        Number of row components.

    n_col_components : int
        Number of col components.

    Returns
    -------
    probability : array, (n_row_components, n_col_components)
    """
    modes = check_array(modes, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(modes, (n_row_components, n_col_components), 'modes')
    return modes

def _check_proportions(proportions, n_row_components, n_col_components):
    """Validate the provided 'means'.

    Parameters
    ----------
    probabilty : array-like, shape (n_row_components, n_col_components)
        The centers of the current components.

    n_row_components : int
        Number of row components.

    n_col_components : int
        Number of col components.

    Returns
    -------
    probability : array, (n_row_components, n_col_components)
    """
    proportions = check_array(proportions, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(proportions, (n_row_components, n_col_components), 'proportions')
    return proportions


###############################################################################
# Latent Block Model parameters estimators (used by the M-Step)

# def compute_modes_proportions(x, tabmu0, tabp0, n_levels, eps=1, iter_max=10):
#   ml = float('-inf')
#   p_ml = tabp0[0]
#   mu_ml = tabmu0[0]
#   
#   n = x.shape[0]
#   w = np.ones(n)
#   _tab_pejs = gettabpej(n_levels)
#   for imu in range(tabmu0.shape[0]):
#       mu = tabmu0[imu]
#       mlold = float('-inf')
#       for ip in range(tabp0.shape[0]):
#           p = tabp0[ip]
#           nostop = 1
#           iter = 0
#           while nostop:
#               iter += 1
#               # -- E step ---
#               # first: compute px for each modality
#               pxi = pallx(_tab_pejs, n_levels, mu, p)
#               px = pxi[x.astype(int)]
#               
#               #vecprod = w * np.log(px)
#               mlnew = np.sum(np.log(px))
#               # first: compute pxz1 for each modality
#               pallxz1 = np.zeros((n_levels, n_levels - 1))
#               
#               for i in range(n_levels):
#                   for j in range(n_levels - 1):
#                     #if n_levels >= 6:
#                     z1tozmm1 = [0 if l != j else 1 for l in range(n_levels -1)]
#                     ivec = [i + 1]
#                     pejv = pej_R.pej(ejVec = ivec, j = n_levels, m = n_levels, mu = int(mu + 1), p = float(p), z1tozjm1Vec = z1tozmm1)
#                     # else:
#                     #   z1tozmm1 = np.array([0 if l != j else 1 for l in range(n_levels -1)])
#                     #   ivec = np.array([i])
#                     #   pejv = pej(ej = ivec, j = n_levels, m = n_levels, mu = mu, p = p, z1tozjm1 = z1tozmm1)
#                   
#                     pallxz1[i, j] = np.maximum(1e-300, pejv) / np.maximum(1e-300, pxi[i])
#               
#               # second: affect each modality value to the corresponding units
#               pxz1 = np.zeros((n, n_levels - 1))
#               
#               for i in range(n_levels):
#                   whereisi = np.where(x == i)[0]
#                   sumwhereisi = whereisi.shape[0]
#                   matsumwhereisi = np.ones((sumwhereisi, 1))
#                   
#                   pallxz1i = pallxz1[i, :]
#                   sub_mat = matsumwhereisi.dot(pallxz1i.reshape(1, -1))
#                   pxz1[whereisi, :] = sub_mat
#               
#               temp1 = np.ones(n_levels - 1)
#               temp2 = w.reshape((-1, 1)).dot(temp1.reshape(1, -1))
#               pxz1 = temp2 * pxz1
#               
#               # ---- M step ----
#               pxz1sum = np.sum(pxz1)
#               pmean = pxz1sum / (n * (n_levels - 1))
#               if mlnew != float('-inf'):
#                   if (np.abs(mlnew - mlold) / n < eps) or (iter > (iter_max - 1)):
#                       nostop = 0
#                       if mlnew > ml:
#                           ml = mlnew
#                           p_ml = pmean
#                           mu_ml = mu
#               else:
#                   if iter > (iter_max - 1):
#                       nostop = 0
#                       if mlnew > ml:
#                           ml = mlnew
#                           p_ml = pmean
#                           mu_ml = mu
#               
#               mlold = mlnew
#   
#   return p_ml, mu_ml.astype(int)
# 
def _estimate_parameters(X, n_levels, proportions_in, resps, opp_resps):
    """Estimate the Gaussian distribution parameters.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input data array.

    resp : array-like, shape (n_samples, n_components)
        The responsibilities for each data sample in X.

    Returns
    -------
    rates
    """
    nk, nl = resps.shape[1], opp_resps.shape[1]
    modes = np.zeros((nk, nl))
    proportions = np.zeros((nk, nl))
    for k in range(nk):
        row_idx = np.where(resps[:, k] == 1)[0]
        X_row = X[row_idx, :]
        for l in range(nl):
            col_idx = np.where(opp_resps[:, l] == 1)[0]
            X_col = X_row[:, col_idx]
            X_col = X_col.reshape(-1)
            #if proportions_in[k, l] < 1e-300:
            # if X.shape[0] < 3000:
            if proportions[k, l] < 1e-3:
              proportion_tab = np.linspace(0.01, 1, 10)
            else:
              proportion_tab = [proportions[k, l]]
            # else:
              # proportion_tab = np.linspace(0, 1, 7)
              #proportion_tab = np.array([proportions_in[k, l]])
            #if X_col.sum() == 0:
            #  modes[k, l] = np.nan
            #  proportions[k, l] = np.nan
            #else:
            mode_tab = np.arange(n_levels)
            emcpp = Bos_utils.Bos_utils()
            proportion, mode = emcpp.ordiemCpp_run(X_col, mode_tab, proportion_tab, _m = n_levels, eps = 1, iter_max = 10)
            # proportion, mode = compute_modes_proportions(X_col, mode_tab, proportion_tab, n_levels)
            modes[k, l] = mode
            proportions[k, l] = proportion
    
    return modes, proportions

###############################################################################
# Latent Block Model probability estimators
class BOSLatentBlockModel(EMfitMMMixin, MixtureModelMixin, BaseEstimator,
                      DensityMixin):
    
    def __init__(self,
                 n_levels,
                 n_row_components=1,
                 n_col_components=1,
                 max_n_steps=200,
                 burn_in_steps=100,
                 resampling_steps=100,
                 resampling_fraction=0.2,
                 abs_tol=1e-9,
                 rel_tol=None,
                 n_init=1,
                 init_method='rand',
                 init_input_method=None,
                 init_resps=None,
                 init_params=None,
                 random_state=None,
                 verbosity=0,
                 history_tracking=0):

        EMfitMMMixin.__init__(self,
                              max_n_steps=max_n_steps,
                              burn_in_steps=burn_in_steps,
                              resampling_steps=resampling_steps,
                              resampling_fraction=resampling_fraction,
                              abs_tol=abs_tol,
                              rel_tol=rel_tol,
                              n_init=n_init,
                              init_method=init_method,
                              init_input_method=init_input_method,
                              init_resps=init_resps,
                              init_params=init_params,
                              random_state=random_state,
                              verbosity=verbosity,
                              history_tracking=history_tracking)
        
        self.n_levels = n_levels
        self.n_row_components = n_row_components
        self.n_col_components = n_col_components

    def _get_parameters(self):
        if hasattr(self, 'removed_mask_'):
            return {'row_weights': self.row_weights_,
                    'col_weights': self.col_weights_,
                    'modes': self.modes_,
                    'proportions': self.proportions_, 
                    'removed_mask': self.removed_mask_}
        else:
            return {'row_weights': self.row_weights_,
                    'col_weights': self.col_weights_,
                    'modes': self.modes_,
                    'proportions': self.proportions_}

    def _set_parameters(self, params):

        if 'row_weights' in params.keys():
            self.row_weights_ = params['row_weights']

        if 'col_weights' in params.keys():
            self.col_weights_ = params['col_weights']

        if 'modes' in params.keys():
            self.modes_ = params['modes']

        if 'proportions' in params.keys():
            self.proportions_ = params['proportions']

        if 'removed_mask' in params.keys():
            self.removed_mask_ = params['removed_mask']
        
        # tot: better job of setting ncomp
        if hasattr(self, 'row_weights_'):
            self.n_row_components = len(self.row_weights_)

        # tot: better job of setting ncomp
        if hasattr(self, 'col_weights_'):
            self.n_col_components = len(self.col_weights_)

    def _get_resps(self):

        return {'row_resps': self.row_resps_,
                'col_resps': self.col_resps_}

    def _set_resps(self, resps):

        if 'row_resps' in resps.keys():
            self.row_resps_ = resps['row_resps']

        if 'col_resps' in resps.keys():
            self.col_resps_ = resps['col_resps']

    def _zero_initialization(self, init_params):
        removed_mask = np.zeros(shape=self.n_row_components).astype(bool)
        init_params['removed_mask'] = removed_mask
        
        return init_params
    
    def _check_parameters(self, X):
        pass

    def _check_clust_param_values(self, X):
        
        params = self._get_parameters()
        
        _check_weights(params['row_weights'], self.n_row_components)

        _check_weights(params['col_weights'], self.n_col_components)

        _check_modes(params['modes'], self.n_row_components, self.n_col_components)

        _check_proportions(params['proportions'], self.n_row_components, self.n_col_components)

    def sample_from_comp(self, y, random_state=None):
        """
        Samples one observation from a cluster.

        Parameters
        ----------
        y: int
            Which cluster to sample from.

        random_state: None, int
            Random seed.

        Output
        ------
        x: array-like, (n_features, )

        """
        y = int(y)
        assert 0 <= y and y < self.n_components

        rng = check_random_state(random_state)

        n_features = self.means_.shape[1]

        # class mean
        m = self.means_[y, :]

        # class covariance
        if self.covariance_type == 'full':
            cov = self.covariances_[y, ...]

        elif self.covariance_type == "tied":
            cov = self.covariances_

        elif self.covariance_type == "diag":
            cov = np.diag(self.covariances_[y, :])

        elif self.covariance_type == "spherical":
            cov = self.covariances_[y] * np.eye(n_features)

        return rng.multivariate_normal(mean=m, cov=cov)

    def comp_row_log_probs(self, X, col_resps):
        n_samples = X.shape[0]
        nk, nl = self.n_row_components, self.n_col_components
        log_probs = np.zeros((n_samples, nk))
        #_tab_pejs = gettabpej(self.n_levels)
        emcpp = Bos_utils.Bos_utils()
        for k in range(nk):
          for l in range(nl):
            pxi = emcpp.pallx_run(self.modes_[k, l], self.proportions_[k, l], self.n_levels)
            pxi = np.array(pxi)
            #pxi = pallx(_tab_pejs, self.n_levels, self.modes_[k, l], self.proportions_[k, l])
            #pallx = comp_probs(self.modes_[k, l], self.proportions_[k, l], self.n_levels)
            px = pxi[X.astype(int)]
            l_prob = np.log(px)
            log_probs[:, k] += np.dot(l_prob, col_resps[:, l])
            
        return log_probs
    
    def comp_col_log_probs(self, X, row_resps):
        n_features = X.shape[1]
        nk, nl = self.n_row_components, self.n_col_components
        log_probs = np.zeros((n_features, nl))
        #_tab_pejs = gettabpej(self.n_levels)
        emcpp = Bos_utils.Bos_utils()
        for l in range(nl):
          for k in range(nk):
            #pallx = comp_probs(self.modes_[k, l], self.proportions_[k, l], self.n_levels)
            #pxi = pallx(_tab_pejs, self.n_levels, self.modes_[k, l], self.proportions_[k, l])
            pxi = emcpp.pallx_run(self.modes_[k, l], self.proportions_[k, l], self.n_levels)
            pxi = np.array(pxi)
            px = pxi[X.astype(int)]
            l_prob = np.log(px).T
            log_probs[:, l] += np.dot(l_prob, row_resps[:, k])
        
        return log_probs

    def _m_step_clust_params(self, X, row_resps, col_resps):
        if hasattr(self, 'proportions_'):
          modes, proportions = \
              _estimate_parameters(X=X, n_levels = self.n_levels, proportions_in = self.proportions_, resps=row_resps, opp_resps=col_resps)
        else:
          proportions_ = np.zeros((self.n_row_components, self.n_col_components))
          modes, proportions = \
            _estimate_parameters(X=X, n_levels = self.n_levels, proportions_in = proportions_, resps=row_resps, opp_resps=col_resps)
        
        return {'modes': modes, 'proportions': proportions}

    def _sm_step_clust_params(self, X, row_resps, col_resps):
        if hasattr(self, 'proportions_'):
          modes, proportions = \
              _estimate_parameters(X=X, n_levels = self.n_levels, proportions_in = self.proportions_, resps=row_resps, opp_resps=col_resps)
        else:
          proportions_ = np.zeros((self.n_row_components, self.n_col_components))
          modes, proportions = \
            _estimate_parameters(X=X, n_levels = self.n_levels, proportions_in = proportions_, resps=row_resps, opp_resps=col_resps)
        
        return {'modes': modes, 'proportions': proportions}

    # def _get_init_clust_parameters(self, X, init_resps, random_state):
    # 
    #     init_params = self._m_step_clust_params(X, np.log(init_resps['row_resps']), np.log(init_resps['col_resps']))
    # 
    #     return init_params
    
    def _get_init_weights(self, X, init_resps, random_state):
      
        init_row_weights = init_resps['row_resps'].sum(0) / init_resps['row_resps'].sum()
        init_col_weights = init_resps['col_resps'].sum(0) / init_resps['col_resps'].sum()

        return init_row_weights, init_col_weights

    def _n_cluster_parameters(self):
        """Return the number of free parameters in the model."""

        n_params = np.sum(self.modes_.shape) + np.sum(self.proportions_.shape)
        
        return int(n_params)

    def _drop_component_params(self, k):
        """
        Drops the cluster parameters of a single componet.
        Subclass should overwrite.
        """
        params = {}
        
        removed_mask = self.removed_mask_
        remaining_idx = np.where(removed_mask == 0)[0]
        remove_idx = remaining_idx[k]
        removed_mask[remove_idx] = True
        params['removed_mask'] = removed_mask        
        
        params['modes'] = np.delete(self.modes_, k, axis=0)
        params['proportions'] = np.delete(self.proportions_, k, axis=0)
       
        return params

    def _reorder_component_params(self, new_idxs):
        """
        Re-orders the component cluster parameters
        """
        params = {}
        params['modes'] = self.modes_[new_idxs, :]
        params['proportions'] = self.proportions_[new_idxs, :]
        return params

    def _combine_SEM_params(self, params_list):
        row_weights_list = []
        col_weights_list = []
        modes_list = []
        proportions_list = []
        if 'removed_mask' in params_list[-1].keys():
            _final_weight_mask = params_list[-1]['removed_mask']
            n_col_components = len(params_list[-1]['col_weights'])
            _final_param_mask = np.tile(_final_weight_mask[:, np.newaxis], n_col_components).reshape(len(_final_weight_mask), n_col_components)
            for params in params_list:
                _input_row_components = params['removed_mask'].shape[0]
                step_row_weights = params['row_weights']
                step_retained_mask = np.invert(params['removed_mask'])
                inter_weights = np.zeros(_input_row_components)
                inter_weights[step_retained_mask] = step_row_weights.reshape(-1)
                inter_weights[_final_weight_mask] = np.zeros(np.sum(_final_weight_mask)).reshape(-1)
                #np.putmask(inter_weights, step_retained_mask, step_row_weights)
                #np.putmask(inter_weights, _final_weight_mask, np.zeros(np.sum(_final_weight_mask)))
                inter_weights /= np.sum(inter_weights)
                row_weights_list.append(inter_weights[np.invert(_final_weight_mask)])
                col_weights_list.append(params['col_weights'])
                
                step_modes = params['modes']
                step_retained_mask = np.invert(params['removed_mask'])
                step_retained_mask = np.tile(step_retained_mask[:, np.newaxis], n_col_components).reshape(len(step_retained_mask), n_col_components)
                
                inter_modes = np.zeros((_input_row_components, n_col_components))
                inter_modes[step_retained_mask] = step_modes.reshape(-1)
                inter_modes[_final_param_mask] = np.zeros(np.sum(_final_param_mask)).reshape(-1)
                #np.putmask(inter_modes, step_retained_mask, step_modes)
                #np.putmask(inter_modes, _final_param_mask, np.zeros(np.sum(_final_param_mask)))
                modes_list.append(inter_modes[np.invert(_final_param_mask)].reshape((self.n_row_components, self.n_col_components)))
                
                step_proportions = params['proportions']
                inter_proportions = np.zeros((_input_row_components, n_col_components))
                inter_proportions[step_retained_mask] = step_proportions.reshape(-1)
                inter_proportions[_final_param_mask] = np.zeros(np.sum(_final_param_mask)).reshape(-1)
                #np.putmask(inter_proportions, step_retained_mask, step_proportions)
                #np.putmask(inter_proportions, _final_param_mask, np.zeros(np.sum(_final_param_mask)))
                proportions_list.append(inter_proportions[np.invert(_final_param_mask)].reshape((self.n_row_components, self.n_col_components)))
        else:
            for params in params_list:
                row_weights_list.append(params['row_weights'])
                col_weights_list.append(params['col_weights'])
                modes_list.append(params['modes'])
                proportions_list.append(params['proportions'])
        
        row_weights = np.average(np.stack(row_weights_list), axis = 0)
        row_weights /= row_weights.sum()
        col_weights = np.average(np.stack(col_weights_list), axis = 0)
        col_weights /= col_weights.sum()
        modes = mode(np.stack(modes_list), keepdims = False)[0]
        proportions = np.average(np.stack(proportions_list), axis = 0)
        params = {'row_weights': row_weights, 'col_weights': col_weights, 
                  'modes': modes, 'proportions': proportions}
        
        return params    

    def computeICL(self, X):
        n_samples, n_features = X.shape   
        nk, nl = self.n_row_components, self.n_col_components
        log_probs = np.zeros((n_samples, n_features, nk, nl))
        emcpp = Bos_utils.Bos_utils()
        for k in range(nk):
          for l in range(nl):
            #pallx = comp_probs(self.modes_[k, l], self.proportions_[k, l], self.n_levels)
            pxi = emcpp.pallx_run(self.modes_[k, l], self.proportions_[k, l], self.n_levels)
            pxi = np.array(pxi)
            px = pxi[X.astype(int)]
            log_probs[:, :, k, l] = np.log(px)
        
        for j in range(n_features):
          for l in range(nl):
            log_probs[:, j, :, l] *= self.row_resps_
        
        for i in range(n_samples):
          for k in range(nk):
            log_probs[i, :, k, :] *= self.col_resps_
        
        return log_probs.sum()
    
    def _get_obs_nll(self, X):
        n_samples, n_features = X.shape   
        nk, nl = self.n_row_components, self.n_col_components
        log_probs = np.zeros((n_samples, n_features, nk, nl))
        emcpp = Bos_utils.Bos_utils()
        for k in range(nk):
          for l in range(nl):
            #pallx = comp_probs(self.modes_[k, l], self.proportions_[k, l], self.n_levels)
            pxi = emcpp.pallx_run(self.modes_[k, l], self.proportions_[k, l], self.n_levels)
            pxi = np.array(pxi)            
            px = pxi[X.astype(int)]
            log_probs[:, :, k, l] = np.log(px)
        
        for j in range(n_features):
          for l in range(nl):
            log_probs[:, j, :, l] *= self.row_resps_
        
        for i in range(n_samples):
          for k in range(nk):
            log_probs[i, :, k, :] *= self.col_resps_

        return (np.sum(self.col_resps_ * np.log(self.col_resps_)) +
          self.col_resps_.sum(0) @ np.log(self.col_weights_).T  + 
          log_probs.sum())

    def _impute_missing_data(self, X, random_state):
        # random_state = check_random_state(random_state)
        # row_idx, col_idx = np.where(self._nan_mask)
        # row_components = np.argmax(self.row_resps_[row_idx, :], 1)
        # col_components = np.argmax(self.col_resps_[col_idx, :], 1)
        # _tab_pejs = gettabpej(self.n_levels)
        # for i in range(row_components.shape[0]):
        #     mode = self.modes_[row_components[i], col_components[i]]
        #     proportion = self.proportions_[row_components[i], col_components[i]]
        #     pxi = pallx(_tab_pejs, self.n_levels, mode, proportion)
        #     X[row_idx[i], col_idx[i]] = random_state.choice(np.arange(self.n_levels), size = 1, p = pxi)
        # 
        random_state = check_random_state(random_state)
        row_idx, col_idx = np.where(self._nan_mask)
        if len(row_idx) > 0:
          row_components = np.argmax(self.row_resps_, 1)
          col_components = np.argmax(self.col_resps_, 1)
          emcpp = Bos_utils.Bos_utils()
          row_component = row_components[row_idx]
          col_component = col_components[col_idx]
          comp_pairs = np.stack(list(zip(row_component, col_component)))
          for comp_pair in np.unique(comp_pairs, axis = 0):
            idx = np.where((comp_pairs == comp_pair).all(axis = 1))[0]
            row_c = comp_pair[0]
            col_c = comp_pair[1]
            mode = self.modes_[row_c, col_c]
            proportion = self.proportions_[row_c, col_c]
            pxi = emcpp.pallx_run(mode, proportion, self.n_levels)
            pxi = [abs(round(i, 10)) for i in pxi]
            X[row_idx[idx], col_idx[idx]] = random_state.choice(np.arange(self.n_levels), size = len(idx), p = pxi)
        
        return X

BOSLatentBlockModel.__doc__ = dedent("""\
LatentBlockModel fit using an EM algorithm.

Parameters
----------
n_components: int
    Number of cluster components.

covariance_type: str
    Type of covariance parameters to use. Must be one of:
    'full'
        Each component has its own general covariance matrix.
    'tied'
        All components share the same general covariance matrix.
    'diag'
        Each component has its own diagonal covariance matrix.
    'spherical'
        Each component has its own single variance.

reg_covar: float
    Non-negative regularization added to the diagonal of covariance.
    Allows to assure that the covariance matrices are all positive.

{em_param_docs}

Attributes
----------

weights_

means_

covariances_

metadata_

""".format(**_em_docs))
