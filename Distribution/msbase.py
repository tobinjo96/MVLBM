"""
This module is a lightly modifed version of sklearn.mixture.GaussianMixture().
"""

import numpy as np

from scipy import linalg
from copy import deepcopy

from sklearn.mixture._base import _check_shape
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import row_norms
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, DensityMixin
from warnings import warn
from textwrap import dedent
from gaussian_mixture import GaussianLatentBlockModel
from poisson_mixture import PoissonLatentBlockModel
from bos_mixture import BOSLatentBlockModel
from base import EMfitMMMixin, MixtureModelMixin, _em_docs

from time import time
###############################################################################
# Latent Block Model probability estimators
class MultiSetBlockModel(EMfitMMMixin, MixtureModelMixin, BaseEstimator,
                      DensityMixin):

    def __init__(self,
                 col_indices,
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
        self.col_indices = col_indices
        self.n_sets = len(col_indices)
    
    def _set_nan_mask(self, X):
        self._nan_mask = np.isnan(X)
        # if hasattr(self, 'view_models_'):
        #     for v in range(self.n_sets):
        #         self.set_models_[v]._set_nan_mask(X[:, self.col_indices[v]])
        # else:
        #     for v in range(self.n_sets):
        #         self.base_set_models[v]._set_nan_mask(X[:, self.col_indices[v]])
        for v in range(self.n_sets):
            self.base_set_models[v]._set_nan_mask(X[:, self.col_indices[v]])
            self.set_models_[v]._set_nan_mask(X[:, self.col_indices[v]])
     
    def _get_parameters(self):
        set_params = []
        col_weights = []
        
        for v in range(self.n_sets):
            params = self.set_models_[v]._get_parameters()
            set_params.append(params)
            col_weights.append(params['col_weights'])
        
        col_weights = np.concatenate(col_weights)
        col_weights /= np.sum(col_weights)
        
        if hasattr(self, 'removed_mask_'):
            return {'set parameters': set_params,
                    'row_weights': self.row_weights_,
                    'col_weights': col_weights, 
                    'removed_mask': self.removed_mask_}
        else:
            return {'set parameters': set_params,
                    'row_weights': self.row_weights_,
                    'col_weights': col_weights}

    def _set_parameters(self, params):
        
        if 'row_weights' in params.keys():
            self.row_weights_ = params['row_weights']
            for v in range(self.n_sets):
                self.set_models_[v].row_weights_ = self.row_weights_
            
        if 'col_weights' in params.keys():
            self.col_weights_ = params['col_weights']
            for v in range(self.n_sets):
                set_col_weights = self.col_weights_[self.col_comp_indices[v]] 
                set_col_weights = set_col_weights / set_col_weights.sum()
                self.set_models_[v].col_weights_ = set_col_weights
        
        if 'set parameters' in params.keys():
            for v in range(self.n_sets):
                self.set_models_[v]._set_parameters(params = params['set parameters'][v])
        
        if 'removed_mask' in params.keys():
            self.removed_mask_ = params['removed_mask']
            for v in range(self.n_sets):
                self.set_models_[v].removed_mask_ = deepcopy(params['removed_mask'])
        
        # tot: better job of setting ncomp
        #if hasattr(self, 'row_weights_'):
        #    self.n_row_components = len(self.row_weights_)

        # tot: better job of setting ncomp
        #if hasattr(self, 'col_weights_'):
        #    self.n_col_components = len(self.col_weights_)

    def _get_resps(self):
        col_resps = np.zeros((self.n_features, self.n_col_components))
        for v in range(self.n_sets):
          mask1 = np.zeros((self.n_features, self.n_col_components), dtype = bool)
          mask2 = np.zeros((self.n_features, self.n_col_components), dtype = bool)
          mask1[self.col_indices[v], :] = True
          mask2[:, self.col_comp_indices[v]] = True
          mask = mask1 * mask2
          values = self.set_models_[v]._get_resps()['col_resps']
          col_resps[mask] = values.reshape(-1)
        
        col_resps = col_resps / col_resps.sum(1)[:, np.newaxis]
        
        return {'row_resps': self.row_resps_,
                'col_resps': col_resps}

    def _set_resps(self, resps):
        #if 'sets' in resps.keys():
        #      for v in range(self.n_sets):
        #          self.set_models_[v]._set_resps(resps['sets'][v])

        if 'row_resps' in resps.keys():
            self.row_resps_ = resps['row_resps']
            for v in range(self.n_sets):
                self.set_models_[v].row_resps_ = self.row_resps_
        
        if 'col_resps' in resps.keys():
            self.col_resps_ = resps['col_resps']
            #set_idx = [0]
            for v in range(self.n_sets):
                #set_idx.append(np.cumsum(self.n_set_col_components)[v])
                set_col_resps_ = self.col_resps_[self.col_indices[v], :][:, self.col_comp_indices[v]]
                set_col_resps_ = set_col_resps_ / set_col_resps_.sum(1)[:, np.newaxis]
                self.set_models_[v].col_resps_ = set_col_resps_ 
                #set_idx = set_idx.pop(0)
    
    def _zero_initialization(self, init_params):
        removed_mask = np.zeros(shape=self.n_row_components).astype(bool)
        init_params['removed_mask'] = removed_mask
        
        for v in range(self.n_sets):
            init_params['set parameters'][v] = self.set_models_[v]._zero_initialization(init_params['set parameters'][v])
        
        return init_params
    
    @property
    def n_set_features(self):
       return [idx[-1] - idx[0] + 1 for idx in self.col_indices]

    @property
    def n_set_col_components(self):
        """
        Number of components in each views.
        """
        if hasattr(self, 'set_models_'):
            return [vm.n_col_components for vm in self.set_models_]
        else:
            return [vm.n_col_components for vm in self.base_set_models]
    
    @property
    def n_col_components(self):
        """
        Number of components in each views.
        """
        return sum(self.n_set_col_components)
    
    @property
    def n_features(self):
        return np.max(np.concatenate(self.col_indices)) + 1
    
    #@property
    #def col_indices(self):
    #    col_indices = []
    #    
    #    for d in range(self.n_sets - 1):
    #      col_indices.append(np.arange(self.idx_list[d], self.idx_list[d+1]))
    #    
    #    col_indices.append(np.arange(self.idx_list[-1], self.n_features))
    #    return col_indices
      
    @property
    def col_comp_indices(self):
        starts_a = [0]
        starts_b = [self.n_set_col_components[i] for i in range(self.n_sets - 1)]
        starts_a.extend(starts_b)
        starts = np.cumsum(starts_a)
        ends = np.cumsum(self.n_set_col_components)
        col_comp_indices = []
        for v in range(self.n_sets):
          col_comp_indices.append(np.arange(starts[v], ends[v]))
        
        return col_comp_indices
        
    def _check_parameters(self, X):
        pass

    def _check_clust_param_values(self, X):
        
        for v in range(self.n_sets):
            self.set_models_[v]._check_clust_param_values(X[:, self.col_indices[v]])

    def _check_fitting_parameters(self, X):
        for v in range(self.n_sets):
            self.base_set_models[v]._check_fitting_parameters(X[:, self.col_indices[v]])

    def _resample_null_col_components(self, resps, random_state):
        random_state = check_random_state(random_state)
        for v in range(self.n_sets):
          set_resps_ = resps[self.col_indices[v], :][:, self.col_comp_indices[v]]
          n_features, n_col_components = set_resps_.shape
          n_resample = np.max((1, np.round(n_features * self.resampling_fraction).astype(int)))
          resampling_index = random_state.choice(range(n_features), size = n_resample, replace = False)
          if n_resample < n_col_components:
            resample_values = random_state.choice(range(n_col_components), size = n_resample)
          else:
            resample_values = np.append(np.arange(n_col_components), random_state.choice(np.arange(n_col_components), size = n_resample - n_col_components))
            random_state.shuffle(resample_values)
          
          resampled_resps = np.zeros((n_resample,n_col_components))
          resampled_resps[np.arange(resample_values.size), resample_values] = 1
          set_resps_[resampling_index, :] = resampled_resps
          mask1 = np.zeros(resps.shape, dtype = bool)
          mask2 = np.zeros(resps.shape, dtype = bool)
          mask1[self.col_indices[v], :] = True
          mask2[:, self.col_comp_indices[v]] = True
          mask = mask1 * mask2
          resps[mask] = set_resps_.reshape(-1)
        
        return resps

    def comp_row_log_probs(self, X, col_resps):
        n_samples, _ = X.shape
        comp_lpr = np.zeros((n_samples, self.n_row_components))

        # log liks for each view's clusters
        # [f(x(v)| theta(v)) for v in n_views)

        set_log_probs = [self.set_models_[v].comp_row_log_probs(X[:, self.col_indices[v]], self.set_models_[v].col_resps_)
                          for v in range(self.n_sets)]

        for k in range(self.n_row_components):
            for v in range(self.n_sets):
                comp_lpr[:, k] += set_log_probs[v][:, k]

        return comp_lpr
    
    def comp_col_log_probs(self, X, row_resps):
        col_log_probs = np.ones((self.n_features, self.n_col_components)) * (-np.inf)
        for v in range(self.n_sets):
          mask1 = np.zeros((self.n_features, self.n_col_components), dtype = bool)
          mask2 = np.zeros((self.n_features, self.n_col_components), dtype = bool)
          mask1[self.col_indices[v], :] = True
          mask2[:, self.col_comp_indices[v]] = True
          mask = mask1 * mask2
          values = self.set_models_[v].comp_col_log_probs(X[:, self.col_indices[v]], row_resps)
          col_log_probs[mask] = values.reshape(-1)
          #np.putmask(col_log_probs, mask, values)
        
        return col_log_probs

    def _m_step_clust_params(self, X, row_resps, col_resps):
        
        params = []
        for v in range(self.n_sets):
          set_col_resps_ = col_resps[self.col_indices[v], :][:, self.col_comp_indices[v]]
          #set_col_resps_ = np.exp(log_set_col_resps_)
          #set_col_resps_ = set_col_resps_ / set_col_resps_.sum(1)[:, np.newaxis]
          #log_set_col_resps_ = np.log(set_col_resps_)
          params.append(self.set_models_[v]._m_step_clust_params(X[:, self.col_indices[v]], row_resps, set_col_resps_))

        return {'set parameters': params}

    def _sm_step_clust_params(self, X, row_resps, col_resps):
        
        params = []
        for v in range(self.n_sets):
          set_col_resps_ = col_resps[self.col_indices[v], :][:, self.col_comp_indices[v]]
          #set_col_resps_ = np.exp(log_set_col_resps_)
          #set_col_resps_ = set_col_resps_ / set_col_resps_.sum(1)[:, np.newaxis]
          #log_set_col_resps_ = np.log(set_col_resps_)
          params.append(self.set_models_[v]._sm_step_clust_params(X[:, self.col_indices[v]], row_resps, set_col_resps_))

        return {'set parameters': params}
    
    def _get_init_weights(self, X, init_resps, random_state):
      
        init_row_weights = init_resps['row_resps'].sum(0) / init_resps['row_resps'].sum()
        init_col_weights = init_resps['col_resps'].sum(0) / init_resps['col_resps'].sum()

        return init_row_weights, init_col_weights

    def _n_cluster_parameters(self):
        """Return the number of free parameters in the model."""
        
        n_cluster_parameters = 0
        for v in range(self.n_sets):
          n_cluster_parameters += self.set_models_[v]._n_cluster_parameters()

        return int(n_cluster_parameters)
    
    def reduce_n_row_components(self):
        print("Reducing")
        self.n_row_components = self.n_row_components -1 
        for v in range(self.n_sets):
            self.set_models_[v].n_row_components = deepcopy(self.n_row_components)
            print(self.set_models_[v].n_row_components)
    
    def _drop_component_params(self, k):
        """
        Drops the cluster parameters of a single componet.
        Subclass should overwrite.
        """
        
        removed_mask = self.removed_mask_
        remaining_idx = np.where(removed_mask == 0)[0]
        remove_idx = remaining_idx[k]
        removed_mask[remove_idx] = True
        
        set_params = []
        for v in range(self.n_sets):
          set_params.append(self.set_models_[v]._drop_component_params(k))
        
        return {'set parameters': set_params, 'removed_mask': removed_mask}

    def _reorder_component_params(self, new_idxs):
        """
        Re-orders the component cluster parameters
        """
        params = []
        for v in range(self.n_sets):
          params.append(self.set_models_[v]._reorder_component_params(new_idxs))
        
        return {'set parameters': params}
    
    def _combine_SEM_params(self, params_list):
        out_params = {}
        row_weights_list = []
        col_weights_list = []
        probability_list = []
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

        else:
            for params in params_list:
                row_weights_list.append(params['row_weights'])
                col_weights_list.append(params['col_weights'])
        
        row_weights = np.average(np.stack(row_weights_list), axis = 0)
        row_weights /= row_weights.sum()
        out_params['row_weights'] = row_weights
        col_weights = np.average(np.stack(col_weights_list), axis = 0)
        col_weights /= col_weights.sum()
        out_params['set parameters'] = []
        out_params['col_weights'] = col_weights
        for v in range(self.n_sets):
            set_params_list = [set_params['set parameters'][v] for set_params in params_list]
            params = self.set_models_[v]._combine_SEM_params(set_params_list)
            out_params['set parameters'].append(params)
        
        return out_params    
# 
#     def _combine_SEM_params(self, params_list):
#         out_params = {}
#         
#         row_weights = [params['row_weights'] for params in params_list]
#         row_weights = np.average(np.stack(row_weights), axis = 0)
#         row_weights /= row_weights.sum()
#         out_params['row_weights'] = row_weights
#         col_weights = [params['col_weights'] for params in params_list]
#         col_weights = np.average(np.stack(col_weights), axis = 0)
#         col_weights /= col_weights.sum()
#         out_params['col_weights'] = col_weights
#         out_params['set parameters'] = []
#         for v in range(self.n_sets):
#           set_params_list = [params['set parameters'][v] for params in params_list]
#           params = self.set_models_[v]._combine_SEM_params(set_params_list)
#           out_params['set parameters'].append(params)
#         
#         return out_params
    
    def computeICL(self, X):
        icl = 0
        for v in range(self.n_sets):
            icl += self.set_models_[v].computeICL(X[:, self.col_indices[v]])
        
        return icl
    
    def _get_obs_nll(self, X):
        obs_nll = 0
        for v in range(self.n_sets):
          obs_nll += self.set_models_[v]._get_obs_nll(X[:, self.col_indices[v]])
        
        return obs_nll
      
    def sample_from_comp(self, y, random_state=None):
        raise NotImplementedError

    # def initialize_parameters(self, X, random_state=None):
    #     random_state = check_random_state(random_state)
    #     
    #     #self.n_features = X.shape[1]
    #     init_resps = self.get_init_resps(X, random_state)
    #     init_params = self._get_init_clust_parameters(X, init_resps, random_state)
    #     init_row_weights, init_col_weights = self._get_init_weights(X, init_resps, random_state)
    #     init_params['row_weights'] = init_row_weights
    #     init_params['col_weights'] = init_col_weights
    #     # over write initialized parameters with user provided parameters
    #     init_params = self._update_user_init(init_params=init_params)
    #     init_params = self._zero_initialization(init_params=init_params)
    #     self._set_parameters(params=init_params)
    #     self._set_resps(resps=init_resps)
    #     self._check_clust_parameters(X)

    def get_init_resps(self, X, random_state):
        #if self.init_resps_method == 'rand_resps':
        n_samples, n_features = X.shape
        #row_resps = random_state.rand(n_samples, self.n_row_components)
        #row_resps /= row_resps.sum(axis=1)[:, np.newaxis]
        row_assign = np.arange(self.n_row_components)
        row_assign = np.append(row_assign, random_state.choice(range(self.n_row_components), size = n_samples - self.n_row_components))
        random_state.shuffle(row_assign)
        row_resps = np.zeros((row_assign.size, row_assign.max() + 1))
        row_resps[np.arange(row_assign.size), row_assign] = 1
        col_resps = np.zeros((n_features, self.n_col_components))
        for v in range(self.n_sets):
          if (isinstance(self.base_set_models[v], GaussianLatentBlockModel) or isinstance(self.base_set_models[v], PoissonLatentBlockModel) or isinstance(self.base_set_models[v], BOSLatentBlockModel))and np.sum(np.isnan(X[:, self.col_indices[v]])) == 0:
            mask1 = np.zeros((n_features, self.n_col_components), dtype = bool)
            mask2 = np.zeros((n_features, self.n_col_components), dtype = bool)
            mask1[self.col_indices[v], :] = True
            mask2[:, self.col_comp_indices[v]] = True
            mask = mask1 * mask2
            kmeans = KMeans(n_clusters=self.n_set_col_components[v], random_state=random_state, n_init=3).fit(X[:, self.col_indices[v]].T)
            col_assign = kmeans.predict(X[:, self.col_indices[v]].T)
            values = np.zeros((col_assign.size, col_assign.max() + 1))
            values[np.arange(col_assign.size), col_assign] = 1
            #values = random_state.rand(self.n_set_features[v], self.n_set_col_components[v])
            col_resps[mask] = values.reshape(-1)
          
          else:
            mask1 = np.zeros((n_features, self.n_col_components), dtype = bool)
            mask2 = np.zeros((n_features, self.n_col_components), dtype = bool)
            mask1[self.col_indices[v], :] = True
            mask2[:, self.col_comp_indices[v]] = True
            mask = mask1 * mask2
            col_assign = np.arange(self.n_set_col_components[v])
            col_assign = np.append(col_assign, random_state.choice(range(self.n_set_col_components[v]), size = self.n_set_features[v] - self.n_set_col_components[v]))
            random_state.shuffle(col_assign)
            values = np.zeros((col_assign.size, col_assign.max() + 1))
            values[np.arange(col_assign.size), col_assign] = 1
            #values = random_state.rand(self.n_set_features[v], self.n_set_col_components[v])
            col_resps[mask] = values.reshape(-1)

        col_resps /= col_resps.sum(axis=1)[:, np.newaxis]
        
        return {'row_resps': row_resps, 'col_resps': col_resps}
      
    def _impute_missing_data(self, X, random_state):
        X_new = np.zeros(X.shape)
        for v in range(self.n_sets):
            X_new[:, self.col_indices[v]] = self.set_models_[v]._impute_missing_data(X[:, self.col_indices[v]], random_state)
        
        return X_new
      
MultiSetBlockModel.__doc__ = dedent("""\
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
