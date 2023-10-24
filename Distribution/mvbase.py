from sklearn.utils import check_random_state
from scipy.special import logsumexp
from sklearn.utils.validation import check_is_fitted
from copy import deepcopy
from time import time
import numpy as np
from itertools import product, chain

from base import MixtureModelMixin, EMfitMMMixin, _check_X

from multiprocessing import Process
from joblib import Parallel, delayed

class MultiViewMixtureModelMixin(MixtureModelMixin):
    
    def _set_nan_mask(self, X):
        for v in range(self.n_views):
          if hasattr(self, 'view_models_'):
            self.view_models_[v]._set_nan_mask(X[v])
          
          self.base_view_models[v]._set_nan_mask(X[v])
    
    def _get_parameters(self):
        view_params = []
        for v in range(self.n_views):
            view_params.append(self.view_models_[v]._get_parameters())

        return {'views': view_params,
                'row_weights': self.row_weights_}

    def _set_parameters(self, params):

        if 'views' in params.keys():
            for v in range(self.n_views):
                self.view_models_[v]._set_parameters(params['views'][v])

        if 'row_weights' in params.keys():
            self.row_weights_ = params['row_weights']

            # make sure each view's weights_ is the marginal of the weights
            #if hasattr(base_final, 'row_resps_') and base_final.row_resps_ is not None:
            for v in range(self.n_views):
                ax_to_sum = tuple([a for a in range(self.n_views) if a != v])
                view_row_weights = np.sum(self.row_weights_mat_, axis=ax_to_sum)
                self.view_models_[v].row_weights_ = view_row_weights
    
    def _get_resps(self):
        view_resps = []
        for v in range(self.n_views):
            view_resps.append(self.view_models_[v]._get_resps())

        return {'views': view_resps,
                'row_resps': self.row_resps_}
    
    def _set_resps(self, resps):
        if 'views' in resps.keys():
              for v in range(self.n_views):
                  self.view_models_[v]._set_resps(resps['views'][v])

        if 'row_resps' in resps.keys():
            self.row_resps_ = resps['row_resps']
            #n_samples = self.row_resps_.shape[0]
            #row_resps_reshape = [n_samples]
            #row_resps_reshape.extend(self.n_view_row_components)
            #row_resps_mat_ = self.row_resps_.reshape(*row_resps_reshape)
            #print(self.row_resps_mat_.shape)
            # make sure each view's weights_ is the marginal of the weights
            for v in range(self.n_views):
                ax_to_sum = tuple([a + 1 for a in range(self.n_views) if a != v])
                view_row_resps = {'row_resps': np.sum(self.row_resps_mat_, axis=ax_to_sum)}
                #view_row_resps = np.sum(row_resps_mat_, axis=ax_to_sum)
                self.view_models_[v]._set_resps(view_row_resps)
                #self.view_models_[v].row_resps_ = view_row_resps
    
    def fit_vem(self, X):
        """
        Fits a multi-view mixture model to the observed multi-view data.

        Parameters
        ----------
        X: list of array-like
            List of data for each view.

        """
        assert len(X) == self.n_views
        
        n_samples = X[0].shape[0]
        
        self.metadata_ = {'n_samples': n_samples,
                          'n_features': [X[v].shape[1] for v in range(self.n_views)]}

        # initalize fitted view models
        self.view_models_ = [deepcopy(self.base_view_models[v])
                             for v in range(self.n_views)]

        for v in range(self.n_views):
          assert X[v].shape[0] == n_samples

          X[v] = _check_X(X[v], self.view_models_[v].n_row_components,self.view_models_[v].n_col_components,
                          ensure_min_samples=2)

          self.view_models_[v]._check_parameters(X[v])
        
        self._check_parameters(X)
        self._check_fitting_parameters(X)

        start_time = time()
        self._fit_vem(X)
        self.metadata_['fit_time'] = time() - start_time
        return self
    
    def fit_sem(self, X):
        """
        Fits a multi-view mixture model to the observed multi-view data.

        Parameters
        ----------
        X: list of array-like
            List of data for each view.

        """
        assert len(X) == self.n_views
        
        n_samples = X[0].shape[0]
        
        self.metadata_ = {'n_samples': n_samples,
                          'n_features': [X[v].shape[1] for v in range(self.n_views)]}

        # initalize fitted view models
        self.view_models_ = [deepcopy(self.base_view_models[v])
                             for v in range(self.n_views)]

        for v in range(self.n_views):
          assert X[v].shape[0] == n_samples

          X[v] = _check_X(X[v], self.view_models_[v].n_row_components,self.view_models_[v].n_col_components,
                          ensure_min_samples=2)

          self.view_models_[v]._check_parameters(X[v])
        
        self._check_parameters(X)
        self._check_fitting_parameters(X)

        start_time = time()
        self._fit_sem(X)
        self.metadata_['fit_time'] = time() - start_time
        return self

    def _check_clust_param_values(self, X):
        """
        Checks cluster parameters.
        """
        for v in range(self.n_views):
            self.view_models_[v]._check_clust_param_values(X[v])

    def _check_clust_parameters(self, X):
        """
        Checks cluster parameters and weights.
        """
        for v in range(self.n_views):
            self.view_models_[v]._check_clust_parameters(X[v])

    @property
    def n_views(self):
        return len(self.base_view_models)

    @property
    def n_view_row_components(self):
        """
        Number of components in each views.
        """
        if hasattr(self, 'view_models_'):
            return [vm.n_row_components for vm in self.view_models_]
        else:
            return [vm.n_row_components for vm in self.base_view_models]

    @property
    def n_row_components(self):
        """
        Returns the total number of clusters.
        """
        # return np.product([self.view_models_[v].n_components
        #                   for v in range(self.n_views)])
        return np.product(self.n_view_row_components)

    @property
    def row_weights_mat_(self):
        """
        Returns weights as a matrix.
        """
        # TODO: does this mess up check_is_fitted()
        if hasattr(self, 'row_weights_') and self.row_weights_ is not None:
            return self.row_weights_.reshape(*self.n_view_row_components)

    @property
    def row_resps_mat_(self):
        """
        Returns resps as a matrix.
        """
        # TODO: does this mess up check_is_fitted()
        n_samples = self.row_resps_.shape[0]
        if self.row_resps_ is not None:
          row_resps_reshape = [n_samples]
          row_resps_reshape.extend(self.n_view_row_components)
          return self.row_resps_.reshape(*row_resps_reshape)

    def comp_row_log_probs(self, X):
        n_samples = X[0].shape[0]
        comp_lpr = np.zeros((n_samples, self.n_row_components))

        # log liks for each view's clusters
        # [f(x(v)| theta(v)) for v in n_views)

        view_log_probs = [self.view_models_[v].comp_row_log_probs(X[v], self.view_models_[v].col_resps_)
                          for v in range(self.n_views)]

        for k in range(self.n_row_components):
            view_idxs = self._get_view_clust_idx(k)

            # f(x| theta_k) = sum_v f(x(v) | theta_k)
            for v in range(self.n_views):
                comp_lpr[:, k] += view_log_probs[v][:, view_idxs[v]]

        return comp_lpr
    
    def comp_col_log_probs(self, X, v):
        
        ax_to_sum = tuple([a + 1 for a in range(self.n_views) if a != v])
        
        view_row_resps = np.sum(self.row_resps_mat_, axis=ax_to_sum)
        
        # log liks for each view's clusters
        # [f(x(v)| theta(v)) for v in n_views)
        
        return self.view_models_[v].comp_col_log_probs(X, view_row_resps)
    
    def sample(self, n_samples=1, random_state=None):
        """
        Generate random samples from the model.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Defaults to 1.

        Returns
        -------
        X : list of array_like, shape (n_samples, n_features)
            List of samples
        """
        #check_is_fitted(self)

        rng = check_random_state(random_state)
        pi = self.weights_
        y_overall = rng.choice(a=np.arange(len(pi)), size=n_samples,
                               replace=True, p=pi)

        samples = [np.zeros((n_samples, self.metadata_['n_features'][v]))
                   for v in range(self.n_views)]

        for i in range(n_samples):
            x = self.sample_from_comp(y=y_overall[i], random_state=rng)
            for v in range(self.n_views):
                samples[v][i, :] = x[v]

        y_views = np.array([self._get_view_clust_idx(y) for y in y_overall])

        return samples, y_overall, y_views

    def sample_from_comp(self, y, random_state=None):
        view_idxs = self._get_view_clust_idx(y)
        return [self.view_models_[v].sample_from_comp(view_idxs[v])
                for v in range(self.n_views)]

    def _n_cluster_parameters(self):
        if hasattr(self, 'view_models_'):
            return sum(vm._n_cluster_parameters()
                       for vm in self.view_models_)

    def _get_view_clust_idx(self, k):
        """Retuns the view cluster indices for each view for an overall cluster index.

        Returns
        -------
        view_idxs : array, shape (n_views, )
        """

        return np.unravel_index(indices=k,
                                shape=self.n_view_row_components,
                                order='C')
        # idx_0, idx_1 = vec2devec_idx(k,
        #                              n_rows=self.n_view_components[0],
        #                              n_cols=self.n_view_components[1])

        # return idx_0, idx_1

    def reorder_components(self, new_idxs):
        raise NotImplementedError

    def drop_component(self, comps):
        raise NotImplementedError

    def _drop_component_params(self, k):
        raise NotImplementedError

    def _reorder_component_params(self, new_idxs):
        raise NotImplementedError

    def bic(self, X):
        """
        Bayesian information criterion for the current model fit
        and the proposed data.

        Parameters
        ----------
        X : array of shape(n_samples, n_dimensions)
        Returns
        -------
        bic: float (the lower the better)
        """
        #check_is_fitted(self)
        n = X[0].shape[0]  # only difference from single view
        return -2 * self.log_likelihood(X) + np.log(n) * self._n_parameters()
    
    def computeICL(self, X):
        n_samples = X[0].shape[0]
        icl = 0
        icl += - (self.n_row_components - 1) / 2 * np.log(n_samples)
        for v in range(self.n_views):
            n_view_features = X[v].shape[1]
            icl += - (self.view_models_[v].n_col_components - 1) / 2 * np.log(n_view_features)
            icl += - self.view_models_[v].n_col_components * self.n_row_components * self.view_models_[v]._n_cluster_parameters() / 2 * np.log(n_samples *n_view_features)
            icl += self.view_models_[v].computeICL(X[v])
            icl += (np.sum(self.view_models_[v].col_resps_, axis = 0) * np.log(self.view_models_[v].col_weights_)).sum()
            #icl += self.view_models_[v].computeICL(X)
        
        
        icl += (np.sum(self.row_resps_, axis = 0) * np.log(self.row_weights_)).sum()
        
        self._icl = icl
        return icl
    
    def predict(self, X):
        y = {}
        y['row_labels'] = self.row_log_probs(X).argmax(axis=1)
        
        y['view_labels'] = [{}] * self.n_views
        
        for v in range(self.n_views):
          y['view_labels'][v]['row_labels'] = np.array([self._get_view_clust_idx(y) for y in y['row_labels']])[:, v]
          y['view_labels'][v]['col_labels'] = self.col_log_probs(X[v], v).argmax(axis=1)
        
        return y
    
    def predict_view_marginal_probas(self, X):
        """
        Parameters
        ----------
        X: list of array-like
            Observed view data.

        Output
        ------
        view_clust_probas: list of array-like
            The vth entry of this list is the
            (n_samples_test, n_view_components[v]) matrix whose (i, k_v)th
            entry is the probability that sample i belongs to view cluster k_v
        """

        p_overall = self.predict_proba(X)
        n_samples = len(p_overall)

        view_clust_probas = [np.zeros((n_samples, self.n_view_components[v]))
                             for v in range(self.n_views)]

        for k in range(self.n_components):
            view_idxs = self._get_view_clust_idx(k)

            for i, v in product(range(n_samples), range(self.n_views)):
                view_clust_probas[v][i, view_idxs[v]] += p_overall[i, k]

        return view_clust_probas


class MultiViewEMixin(EMfitMMMixin):
    
    def initialize_parameters(self, X, random_state=None):
        """
        Parameters
        ----------
        X: list of array-like
            List of data for each view.

        #random_state: int, None
        #    Random seed for initializations.
        """

        random_state = check_random_state(random_state)
        
        n_samples, n_features = X[0].shape
        
        self._set_nan_mask(X)
        
        ##############################
        # initialize view parameters #
        ##############################

        init_view_params = []
        init_view_resps = []
        
        if self.init_method == 'user':
            if self.init_input_method == 'resps':
              for v in range(self.n_views):
                vm = deepcopy(self.base_view_models[v])
                vm._set_resps(resps=self.init_resps['views'][v])
                init_row_weights, init_col_weights = vm._get_init_weights(X[v], self.init_resps['views'][v], random_state)
                X[v] = vm._sampling_imputation_initialization(X[v], self.init_resps['views'][v], random_state)
                init_params = vm._get_init_clust_parameters(X[v], self.init_resps['views'][v], random_state)
                init_params['row_weights'] = init_row_weights
                init_params['col_weights'] = init_col_weights
                # over write initialized parameters with user provided parameters
                init_params = vm._update_user_init(init_params=init_params)
                init_params = vm._zero_initialization(init_params=init_params)
                vm._set_parameters(params=init_params)
                vm._check_clust_parameters(X[v])
                init_view_params.append(vm._get_parameters())
                init_view_resps.append(vm._get_resps())
               
              if 'row_resps' in self.init_resps.keys():
                init_row_resps = self.init_resps['row_resps']
                init_row_weights = init_row_resps.sum(0)
              
              else:
                init_row_weights = np.repeat(1 / self.n_row_components, self.n_row_components)
                row_resps_mat_shape = [n_samples]
                row_resps_mat_shape.extend(self.n_view_row_components)
                init_row_resps_mat = np.zeros(row_resps_mat_shape)
                for v in range(self.n_views):
                  not_v_idx = [s for s in range(self.n_views) if s != v]
                  init_view_resps_split = init_view_resps[v]['row_resps']
                  not_v_idx = [i + 1 for i in not_v_idx]
                  init_view_resps_split = np.expand_dims(init_view_resps_split, not_v_idx)
                  init_view_resps_split = np.broadcast_arrays(init_view_resps_split, init_row_resps_mat)[0]
                  init_row_resps_mat += init_view_resps_split
                
                init_row_resps = init_row_resps_mat.reshape((n_samples, self.n_row_components))
                init_row_resps = init_row_resps / init_row_resps.sum(1)[:, np.newaxis]
                row_assign = (init_row_resps.cumsum(1) > random_state.rand(init_row_resps.shape[0])[:,None]).argmax(1)
                init_row_resps = np.zeros((n_samples, self.n_row_components))
                init_row_resps[np.arange(row_assign.size), row_assign] = 1
            
            elif self.init_input_method == 'params':
              for v in range(self.n_views):
                vm = deepcopy(self.base_view_models[v])
                vm.initialize_parameters(X[v], random_state=random_state)
                init_view_params.append(vm._get_parameters())
                init_view_resps.append(vm._get_resps())
                 
              init_row_weights = np.repeat(1 / self.n_row_components, self.n_row_components)
              row_resps_mat_shape = [n_samples]
              row_resps_mat_shape.extend(self.n_view_row_components)
              init_row_resps_mat = np.zeros(row_resps_mat_shape)
              for v in range(self.n_views):
                not_v_idx = [s for s in range(self.n_views) if s != v]
                init_view_resps_split = init_view_resps[v]['row_resps']
                not_v_idx = [i + 1 for i in not_v_idx]
                init_view_resps_split = np.expand_dims(init_view_resps_split, not_v_idx)
                init_view_resps_split = np.broadcast_arrays(init_view_resps_split, init_row_resps_mat)[0]
                init_row_resps_mat += init_view_resps_split
              
              init_row_resps = init_row_resps_mat.reshape((n_samples, self.n_row_components))
              init_row_resps = init_row_resps / init_row_resps.sum(1)[:, np.newaxis]
              row_assign = (init_row_resps.cumsum(1) > random_state.rand(init_row_resps.shape[0])[:,None]).argmax(1)
              init_row_resps = np.zeros((n_samples, self.n_row_components))
              init_row_resps[np.arange(row_assign.size), row_assign] = 1
            
            else:
              raise ValueError('Bad Input for init_input_method.')
        
        elif self.init_method == 'rand':
            for v in range(self.n_views):
              vm = deepcopy(self.base_view_models[v])
              vm.initialize_parameters(X[v], random_state=random_state)
              init_view_params.append(vm._get_parameters())
              init_view_resps.append(vm._get_resps())
                              
            init_row_weights = np.repeat(1 / self.n_row_components, self.n_row_components)
            row_resps_mat_shape = [n_samples]
            row_resps_mat_shape.extend(self.n_view_row_components)
            init_row_resps_mat = np.zeros(row_resps_mat_shape)
            for v in range(self.n_views):
              not_v_idx = [s for s in range(self.n_views) if s != v]
              init_view_resps_split = init_view_resps[v]['row_resps']
              not_v_idx = [i + 1 for i in not_v_idx]
              init_view_resps_split = np.expand_dims(init_view_resps_split, not_v_idx)
              init_view_resps_split = np.broadcast_arrays(init_view_resps_split, init_row_resps_mat)[0]
              init_row_resps_mat += init_view_resps_split
            
            init_row_resps = init_row_resps_mat.reshape((n_samples, self.n_row_components))
            init_row_resps = init_row_resps / init_row_resps.sum(1)[:, np.newaxis]
            row_assign = (init_row_resps.cumsum(1) > random_state.rand(init_row_resps.shape[0])[:,None]).argmax(1)
            init_row_resps = np.zeros((n_samples, self.n_row_components))
            init_row_resps[np.arange(row_assign.size), row_assign] = 1
        
        else:
            raise ValueError('Bad Input for init_method.')
        
        init_params = {'views': init_view_params, 'row_weights': init_row_weights}

        init_params = self._zero_initialization(init_params=init_params)
        #self._set_nan_mask(X)

        self._set_parameters(params=init_params)
        init_resps = {'views': init_view_resps, 'row_resps': init_row_resps}
        
        self._set_resps(resps=init_resps)
        self._check_clust_parameters(X)

    def _get_init_clust_parameters(self, X, random_state):
        # Not used by multi-view model
        raise NotImplementedError

    def _get_init_weights(self, X, random_state):
        # Not used by multi-view model
        raise NotImplementedError

    def _update_user_init(self, init_params=None, init_weights=None):
        # Not used by multi-view model
        raise NotImplementedError
    
    def _check_fitting_parameters(self, X):
        for v in range(self.n_views):
            self.base_view_models[v]._check_fitting_parameters(X[v])
    
    def _combine_SEM_params(self, params_list):
        out_params = {}
        
        row_weights = [params['row_weights'] for params in params_list]
        row_weights = np.average(np.stack(row_weights), axis = 0)
        row_weights /= row_weights.sum()
        out_params['row_weights'] = row_weights
        
        out_params['views'] = []
        for v in range(self.n_views):
          view_params_list = [params['views'][v] for params in params_list]
          params = self.view_models_[v]._combine_SEM_params(view_params_list)
          out_params['views'].append(params)
        
        return out_params
    
    def _row_m_step_clust_params(self, X, resps):
        """
        M step. Each view's cluster parameters can be updated independently.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        resps_mat = resps.reshape(-1, *self.n_view_row_components)

        view_params = [None for v in range(self.n_views)]
        for v in range(self.n_views):

            # sum over other views
            # note samples are on the first axis hence the +1
            axes2sum = set(range(1, self.n_views + 1))
            axes2sum = list(axes2sum.difference([v + 1]))
            view_resps = np.apply_over_axes(np.sum,
                                              a=resps_mat,
                                              axes=axes2sum)
            # TODO: not sure why the argument to squeeze
            view_resps = view_resps.squeeze(axis=tuple(axes2sum))
            view_params[v] = self.view_models_[v].\
                _m_step_clust_params(X[v], view_resps, self.view_models_[v].col_resps_)

        return view_params
    
    def _col_m_step_clust_params(self, X, resps, v):
        ax_to_sum = tuple([a + 1 for a in range(self.n_views) if a != v])
        view_row_resps = np.sum(self.row_resps_mat_, axis=ax_to_sum)
        
        return self.view_models_[v].\
                _m_step_clust_params(X=X, row_resps=view_row_resps, col_resps=resps)
    
    def _row_sm_step_clust_params(self, X, resps):
        """
        M step. Each view's cluster parameters can be updated independently.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        resps_mat = resps.reshape(-1, *self.n_view_row_components)

        view_params = [None for v in range(self.n_views)]
        for v in range(self.n_views):

            # sum over other views
            # note samples are on the first axis hence the +1
            axes2sum = set(range(1, self.n_views + 1))
            axes2sum = list(axes2sum.difference([v + 1]))
            view_resps = np.apply_over_axes(np.sum,
                                              a=resps_mat,
                                              axes=axes2sum)
            # TODO: not sure why the argument to squeeze
            view_resps = view_resps.squeeze(axis=tuple(axes2sum))
            view_params[v] = self.view_models_[v].\
                _sm_step_clust_params(X[v], view_resps, self.view_models_[v].col_resps_)

        return view_params
    
    def _col_sm_step_clust_params(self, X, resps, v):
        ax_to_sum = tuple([a + 1 for a in range(self.n_views) if a != v])
        view_row_resps = np.sum(self.row_resps_mat_, axis=ax_to_sum)
        
        return self.view_models_[v].\
                _sm_step_clust_params(X=X, row_resps=view_row_resps, col_resps=resps)

    def _row_m_step(self, X, E_out):
        resps = E_out['resps']

        view_params = self._row_m_step_clust_params(X=X, resps=resps)
        row_weights = self._row_m_step_weights(X=X, resps=resps)
        return {'views': view_params, 'row_weights': row_weights}

    def _col_m_step(self, X, E_out, v):
        resps = E_out['resps']

        new_params = self._col_m_step_clust_params(X=X, resps=resps, v=v)
        new_params['col_weights'] = self._col_m_step_weights(X=X, resps=resps)

        return new_params
    
    def _row_sm_step(self, X, E_out):
        resps = E_out['resps']

        view_params = self._row_sm_step_clust_params(X=X, resps=resps)
        row_weights = self._row_sm_step_weights(X=X, resps=resps)
        return {'views': view_params, 'row_weights': row_weights}

    def _col_sm_step(self, X, E_out, v):
        resps = E_out['resps']

        new_params = self._col_sm_step_clust_params(X=X, resps=resps, v=v)
        new_params['col_weights'] = self._col_sm_step_weights(X=X, resps=resps)

        return new_params
    
    def _impute_missing_data(self, X, random_state):
        X_new = []
        for v in range(self.n_views):
            X_new.append(self.view_models_[v]._impute_missing_data(X[v], random_state))
        
        return X_new
    
    
