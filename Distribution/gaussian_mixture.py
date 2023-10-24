"""
This module is a lightly modifed version of sklearn.mixture.GaussianMixture().
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from scipy import linalg

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

def _check_means(means, n_row_components, n_col_components):
    """Validate the provided 'means'.

    Parameters
    ----------
    means : array-like, shape (n_components, n_features)
        The centers of the current components.

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    means : array, (n_components, n_features)
    """
    means = check_array(means, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(means, (n_row_components, n_col_components), 'means')
    return means

def _check_precision_positivity(precision):
    """Check a precision vector is positive-definite."""
    if np.any(np.less_equal(precision, 0.0)):
        raise ValueError("'Precision' should be "
                         "positive" )

def _check_precisions(precisions, n_row_components, n_col_components):
    _check_precision_positivity(precisions)
    _check_shape(precisions, (n_row_components, n_col_components), 'precisions')
    return precisions

def compute_precisions(variances):
    return 1/variances

###############################################################################
#Gaussian mixture parameters estimators (used by the M-Step)
def _estimate_parameters(X, row_resps, col_resps, reg_covar):
    nk, nl = row_resps.shape[1], col_resps.shape[1]
    means = row_resps.T.dot(X).dot(col_resps) / (row_resps.sum(0).reshape(nk, 1) * col_resps.sum(0).reshape(1, nl))
    variances = np.empty((nk, nl))
    for i in range(nk):
      for j in range(nl):
        X_meansub = X - means[i, j]
        variances[i, j] = row_resps[:, i, np.newaxis].T.dot(X_meansub ** 2).dot(col_resps[:, j]) / (row_resps[:, i].sum() * col_resps[:, j].sum())

    precisions = 1/variances
    return nk, means, variances, precisions


###############################################################################
# Latent Block Model probability estimators
class GaussianLatentBlockModel(EMfitMMMixin, MixtureModelMixin, BaseEstimator,
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
                 reg_covar = 1e-6,
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
        self.reg_covar = reg_covar
    
    def _get_parameters(self):
        if hasattr(self, 'removed_mask_'):
            return {'row_weights': self.row_weights_,
                  'col_weights': self.col_weights_,
                  'means': self.means_,               
                  'variances': self.variances_,
                  'precisions': self.precisions_,
                  'removed_mask': self.removed_mask_}
        else:
            return {'row_weights': self.row_weights_,
                  'col_weights': self.col_weights_,
                  'means': self.means_,               
                  'variances': self.variances_,
                  'precisions': self.precisions_
                  }

    def _set_parameters(self, params):

        if 'row_weights' in params.keys():
            self.row_weights_ = params['row_weights']

        if 'col_weights' in params.keys():
            self.col_weights_ = params['col_weights']

        if 'means' in params.keys():
            self.means_ = params['means']

        if 'variances' in params.keys():
            self.variances_ = params['variances']

        if 'precisions' in params.keys():
            self.precisions_ = params['precisions']
        elif 'variances' in params.keys():
            self.precisions_ = compute_precisions(variances = self.variances_)
        
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

        _check_means(params['means'], self.n_row_components, self.n_col_components)
        
        _check_precisions(params['precisions'], self.n_row_components, self.n_col_components)
    
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
        #print("Gaussian Row Log Probs")
        n_samples, n_features = X.shape
        
        nk, nl = self.n_row_components, self.n_col_components
        log_probs = np.zeros((n_samples, nk))
        for k in range(nk):
          for l in range(nl):
           
            l_prob = ((X - self.means_[k, l])**2)/self.variances_[k, l]
            #l_prob = self.means_[k, l]**2 * self.precisions_[k, l] - 2 * np.dot(X, self.means_[k, l] * self.precisions_[k, l]) + np.outer(np.sqrt((X*X).sum(1)), self.precisions_[k, l])
            l_probs = -.5 * (np.log(self.variances_[k, l]) + l_prob )
            
            log_probs[:, k] += np.dot(l_probs, col_resps[:, l])
            
        #print(np.exp(log_probs))       
        return log_probs
    
    def comp_col_log_probs(self, X, row_resps):
        #print("Gaussian Col Log Probs")
        n_samples, n_features = X.shape
        
        nk, nl = self.n_row_components, self.n_col_components
        log_probs = np.zeros((n_features, nl))
        for l in range(nl):
          
          for k in range(nk):
           
            l_prob = ((X - self.means_[k, l])**2)/self.variances_[k, l]
            #l_prob = self.means_[k, l]**2 * self.precisions_[k, l] - 2 * np.dot(X, self.means_[k, l] * self.precisions_[k, l]) + np.outer(np.sqrt((X*X).sum(1)), self.precisions_[k, l])
            
            l_probs = -.5 * (np.log(self.variances_[k, l]) + l_prob.T )
            
            log_probs[:, l] += np.dot(l_probs, row_resps[:, k])
        #print(np.exp(log_probs))        
        return log_probs

    def _m_step_clust_params(self, X, row_resps, col_resps):
        #print("Gaussian SM Step")
        nk, means, variances, precisions = \
            _estimate_parameters(X=X, row_resps=row_resps, col_resps=col_resps, reg_covar = self.reg_covar)
        
        return {'means': means, 'variances': variances, 'precisions': precisions}
    
    def _sm_step_clust_params(self, X, row_resps, col_resps):
        #print("Gaussian SM Step")
        nk, means, variances, precisions = \
            _estimate_parameters(X=X, row_resps=row_resps, col_resps=col_resps, reg_covar = self.reg_covar)
        
        return {'means': means, 'variances': variances, 'precisions': precisions}

    def get_init_resps(self, X, random_state):
        #if self.init_resps_method == 'rand_resps':
        kmeans = KMeans(n_clusters=self.n_row_components, random_state=random_state, n_init=10).fit(X)
        row_assign = kmeans.predict(X)
        row_resps = np.zeros((row_assign.size, row_assign.max() + 1))
        row_resps[np.arange(row_assign.size), row_assign] = 1
        kmeans = KMeans(n_clusters=self.n_col_components, random_state=random_state, n_init=10).fit(X.T)
        col_assign = kmeans.predict(X.T)
        col_resps = np.zeros((col_assign.size, col_assign.max() + 1))
        col_resps[np.arange(col_assign.size), col_assign] = 1
        # else:
        #     raise ValueError("Invalid value for 'init_params_method': {}"
        #                      "".format(self.init_params_method))
        # 
        return {'row_resps': row_resps, 'col_resps': col_resps}

    def _get_init_clust_parameters(self, X, init_resps, random_state):

        init_params = self._sm_step_clust_params(X, init_resps['row_resps'],init_resps['col_resps'])

        return init_params
    
    def _get_init_weights(self, X, init_resps, random_state):
      
        init_row_weights = init_resps['row_resps'].sum(0) / init_resps['row_resps'].sum()
        init_col_weights = init_resps['col_resps'].sum(0) / init_resps['col_resps'].sum()

        return init_row_weights, init_col_weights

    def _n_cluster_parameters(self):
        """Return the number of free parameters in the model."""
        n_params = np.sum(self.means_.shape) + np.sum(self.variances_.shape)
        
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
        
        params['means'] = np.delete(self.means_, k, axis = 0)
        params['variances'] = np.delete(self.variances_, k, axis = 0)
        
        return params

    def _reorder_component_params(self, new_idxs):
        """
        Re-orders the component cluster parameters
        """
        params = {}
        params['means'] = self.means_[new_idxs, :]
        params['variances'] = self.variances_[new_idxs, ...]
        if hasattr(self, 'precisions_'):
            params['precisions_'] = self.precisions_[new_idxs, ...]
        return params
    
    def _combine_SEM_params(self, params_list):
        row_weights_list = []
        col_weights_list = []
        means_list = []
        variances_list = []
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
                
                step_means = params['means']
                step_retained_mask = np.invert(params['removed_mask'])
                step_retained_mask = np.tile(step_retained_mask[:, np.newaxis], n_col_components).reshape(len(step_retained_mask), n_col_components)
                
                inter_means = np.zeros((_input_row_components, n_col_components))
                inter_means[step_retained_mask] = step_means.reshape(-1)
                inter_means[_final_param_mask] = np.zeros(np.sum(_final_param_mask)).reshape(-1)
                #np.putmask(inter_means, step_retained_mask, step_means)
                #np.putmask(inter_means, _final_param_mask, np.zeros(np.sum(_final_param_mask)))
                means_list.append(inter_means[np.invert(_final_param_mask)].reshape((self.n_row_components, self.n_col_components)))
                
                step_variances = params['variances']
                inter_variances = np.zeros((_input_row_components, n_col_components))
                inter_variances[step_retained_mask] = step_variances.reshape(-1)
                inter_variances[_final_param_mask] = np.zeros(np.sum(_final_param_mask)).reshape(-1)
                #np.putmask(inter_variances, step_retained_mask, step_variances)
                #np.putmask(inter_variances, _final_param_mask, np.zeros(np.sum(_final_param_mask)))
                variances_list.append(inter_variances[np.invert(_final_param_mask)].reshape((self.n_row_components, self.n_col_components)))
        else:
            for params in params_list:
                row_weights_list.append(params['row_weights'])
                col_weights_list.append(params['col_weights'])
                means_list.append(params['means'])
                variances_list.append(params['variances'])
        
        row_weights = np.average(np.stack(row_weights_list), axis = 0)
        row_weights /= row_weights.sum()
        col_weights = np.average(np.stack(col_weights_list), axis = 0)
        col_weights /= col_weights.sum()
        means = np.average(np.stack(means_list), axis = 0)
        variances = np.average(np.stack(variances_list), axis = 0)
        params = {'row_weights': row_weights, 'col_weights': col_weights, 
                  'means': means, 'variances': variances}
        
        return params    
    
    def computeICL(self, X):
        n_samples, n_features = X.shape   
        nk, nl = self.n_row_components, self.n_col_components
        
        log_probs = np.zeros((n_samples, n_features, nk, nl))
        #log_probs = np.empty((n_samples, nk))
        for k in range(nk):
          
          #l_probs = np.empty((n_samples, nl, n_features))
          
          for l in range(nl):
           
            l_prob = (X - self.means_[k, l])**2/self.variances_[k, l]
           
            log_probs[:, :, k, l] = -.5 * (np.log(2 * np.pi * self.variances_[k, l]) + l_prob )
        
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
        #ax_to_sum = tuple([a + 1 for a in range(self.n_views) if a != v])
        #view_row_resps = np.sum(self.row_resps_mat_, axis=ax_to_sum)
        
        log_probs = np.zeros((n_samples, n_features, nk, nl))
        #log_probs = np.empty((n_samples, nk))
        for k in range(nk):
          
          #l_probs = np.empty((n_samples, nl, n_features))
          
          for l in range(nl):
           
            l_prob = (X - self.means_[k, l])**2/self.variances_[k, l]
           
            log_probs[:, :, k, l] = -.5 * (np.log(2 * np.pi * self.variances_[k, l]) + l_prob )
        
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
        # #print("Imputing Gauss")
        # random_state = check_random_state(random_state)
        # row_idx, col_idx = np.where(self._nan_mask)
        # row_components = np.argmax(self.row_resps_[row_idx, :], 1)
        # col_components = np.argmax(self.col_resps_[col_idx, :], 1)
        # #components = np.array(list(zip(row_components,col_components)))
        # for i in range(row_components.shape[0]):
        #     mean = self.means_[row_components[i], col_components[i]]
        #     variance = self.variances_[row_components[i], col_components[i]]
        #     #print(mean)
        #     #print(variance)
        #     #print(row_idx[i], col_idx[i])
        #     X[row_idx[i], col_idx[i]] = random_state.normal(loc = mean, scale = np.sqrt(variance), size = 1)
        #     #print(random_state.normal(loc = mean, scale = np.sqrt(variance), size = 1))
        # 
        random_state = check_random_state(random_state)
        row_idx, col_idx = np.where(self._nan_mask)
        if len(row_idx) > 0:
          row_components = np.argmax(self.row_resps_, 1)
          col_components = np.argmax(self.col_resps_, 1)
          row_component = row_components[row_idx]
          col_component = col_components[col_idx]
          comp_pairs = np.stack(list(zip(row_component, col_component)))
          for comp_pair in np.unique(comp_pairs, axis = 0):
            idx = np.where((comp_pairs == comp_pair).all(axis = 1))[0]
            row_c = comp_pair[0]
            col_c = comp_pair[1]
            mean = self.means_[row_c, col_c]
            variance = self.variances_[row_c, col_c]
            X[row_idx[idx], col_idx[idx]] = random_state.normal(loc = mean, scale = np.sqrt(variance), size = len(idx))
        
        return X

GaussianLatentBlockModel.__doc__ = dedent("""\
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
