"""
This module is a lightly modifed version of sklearn.mixture.GaussianMixture().
"""

import numpy as np

from scipy import linalg
from scipy.special import factorial
from sklearn.mixture._base import _check_shape
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import row_norms
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, DensityMixin
from warnings import warn
from textwrap import dedent

from base import EMfitMMMixin, MixtureModelMixin, _em_docs


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


def _check_rates(rates, n_row_components, n_col_components):
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
    rates = check_array(rates, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(rates, (n_row_components, n_col_components), 'rates')
    return rates


###############################################################################
# Latent Block Model parameters estimators (used by the M-Step)

def _estimate_parameters(X, resps, opp_resps):
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
    nk_dot = resps.T.dot(X).sum(1)
    nl_dot = X.dot(opp_resps).sum(0)
    rates = resps.T.dot(X).dot(opp_resps)
    rates = 1/np.outer(nk_dot, nl_dot) * rates
    return rates

###############################################################################
# Latent Block Model probability estimators
class PoissonLatentBlockModel(EMfitMMMixin, MixtureModelMixin, BaseEstimator,
                      DensityMixin):

    def __init__(self,
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

        self.n_row_components = n_row_components
        self.n_col_components = n_col_components

    def _get_parameters(self):
        if hasattr(self, 'removed_mask_'):
          return {'row_weights': self.row_weights_,
                'col_weights': self.col_weights_,
                'rates': self.rates_, 
                'removed_mask': self.removed_mask_}
        else:
          return {'row_weights': self.row_weights_,
                'col_weights': self.col_weights_,
                'rates': self.rates_}
        
    def _set_parameters(self, params):

        if 'row_weights' in params.keys():
            self.row_weights_ = params['row_weights']

        if 'col_weights' in params.keys():
            self.col_weights_ = params['col_weights']

        if 'rates' in params.keys():
            self.rates_ = params['rates']
        
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

        _check_rates(params['rates'], self.n_row_components, self.n_col_components)

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
        ni_dot = X.sum(1)
        nj_dot = X.sum(0)
        ninj = np.outer(ni_dot, nj_dot)
        log_probs = np.zeros((n_samples, nk))
        for k in range(nk):
          for l in range(nl):
            
            l_prob = X * np.log(ninj * self.rates_[k, l]) - ninj * self.rates_[k, l]  - np.log(factorial(X))
            log_probs[:, k] += np.dot(l_prob, col_resps[:, l])
            
        return log_probs
    
    def comp_col_log_probs(self, X, row_resps):

        n_features = X.shape[1]
        nk, nl = self.n_row_components, self.n_col_components
        ni_dot = X.sum(1)
        nj_dot = X.sum(0)
        ninj = np.outer(ni_dot, nj_dot)
        log_probs = np.zeros((n_features, nl))
        for l in range(nl):
          for k in range(nk):
            
            l_prob = X * np.log(ninj * self.rates_[k, l]) - ninj * self.rates_[k, l]  - np.log(factorial(X))
            log_probs[:, l] += np.dot(l_prob.T, row_resps[:, k])
        
        return log_probs

    def _m_step_clust_params(self, X, row_resps, col_resps):
        
        rates = \
            _estimate_parameters(X=X, resps=row_resps, opp_resps=col_resps)
        
        return {'rates': rates}

    def _sm_step_clust_params(self, X, row_resps, col_resps):
        
        rates = \
            _estimate_parameters(X=X, resps=row_resps, opp_resps=col_resps)
        
        return {'rates': rates}

    def _get_init_clust_parameters(self, X, init_resps, random_state):

        init_params = self._m_step_clust_params(X, np.log(init_resps['row_resps']), np.log(init_resps['col_resps']))

        return init_params
    
    def _get_init_weights(self, X, init_resps, random_state):
      
        init_row_weights = init_resps['row_resps'].sum(0) / init_resps['row_resps'].sum()
        init_col_weights = init_resps['col_resps'].sum(0) / init_resps['col_resps'].sum()

        return init_row_weights, init_col_weights

    def _n_cluster_parameters(self):
        """Return the number of free parameters in the model."""

        rate_params = np.sum(self.rates_.shape)
        
        return int(rate_params)

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
        
        params['rates'] = np.delete(self.rates_, k, axis=0)
        
        return params

    def _reorder_component_params(self, new_idxs):
        """
        Re-orders the component cluster parameters
        """
        params = {}
        params['rates'] = self.rates_[new_idxs, :]
        return params
    
    def _combine_SEM_params(self, params_list):
        row_weights_list = []
        col_weights_list = []
        rates_list = []
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
                
                step_rates = params['rates']
                step_retained_mask = np.invert(params['removed_mask'])
                step_retained_mask = np.tile(step_retained_mask[:, np.newaxis], n_col_components).reshape(len(step_retained_mask), n_col_components)
                
                inter_rates = np.zeros((_input_row_components, n_col_components))
                inter_rates[step_retained_mask] = step_rates.reshape(-1)
                inter_rates[_final_param_mask] = np.zeros(np.sum(_final_param_mask)).reshape(-1)
                #np.putmask(inter_rates, step_retained_mask, step_rates)
                #np.putmask(inter_rates, _final_param_mask, np.zeros(np.sum(_final_param_mask)))
                rates_list.append(inter_rates[np.invert(_final_param_mask)].reshape((self.n_row_components, self.n_col_components)))
        else:
            for params in params_list:
                row_weights_list.append(params['row_weights'])
                col_weights_list.append(params['col_weights'])
                rates_list.append(params['rates'])
        
        row_weights = np.average(np.stack(row_weights_list), axis = 0)
        row_weights /= row_weights.sum()
        col_weights = np.average(np.stack(col_weights_list), axis = 0)
        col_weights /= col_weights.sum()
        rates = np.average(np.stack(rates_list), axis = 0)
        params = {'row_weights': row_weights, 'col_weights': col_weights, 
                  'rates': rates}
        
        return params
    
    def computeICL(self, X):
        n_samples, n_features = X.shape   
        nk, nl = self.n_row_components, self.n_col_components
        ni_dot = X.sum(1)
        nj_dot = X.sum(0)
        ninj = np.outer(ni_dot, nj_dot)
        log_probs = np.zeros((n_samples, n_features, nk, nl))
        for k in range(nk):
          
          for l in range(nl):
            
            log_probs[:, :, k, l] = X * np.log(ninj * self.rates_[k, l]) - ninj * self.rates_[k, l]  - np.log(factorial(X))
        
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
        ni_dot = X.sum(1)
        nj_dot = X.sum(0)
        ninj = np.outer(ni_dot, nj_dot)
        log_probs = np.zeros((n_samples, n_features, nk, nl))
        for k in range(nk):
          
          for l in range(nl):
            
            log_probs[:, :, k, l] = X * np.log(ninj * self.rates_[k, l]) - ninj * self.rates_[k, l]  - np.log(factorial(X))
        
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
        random_state = check_random_state(random_state)
        row_idx, col_idx = np.where(self._nan_mask)
        row_components = np.argmax(self.row_resps_[row_idx, :], 1)
        col_components = np.argmax(self.col_resps_[col_idx, :], 1)
        ni_dot = X.sum(1)
        nj_dot = X.sum(0)
        ninj = np.outer(ni_dot, nj_dot)
        for i in range(row_components.shape[0]):
            rate = self.rates_[row_components[i], col_components[i]]
            lam = rate * ninj[row_idx[i], col_idx[i]]
            X[row_idx[i], col_idx[i]] = random_state.poisson(lam = lam, size = 1)
        
        return X

PoissonLatentBlockModel.__doc__ = dedent("""\
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
