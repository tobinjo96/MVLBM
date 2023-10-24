from abc import ABCMeta, abstractmethod
from sklearn.utils import check_random_state, check_array
# from sklearn.base import BaseEstimator, DensityMixin
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_is_fitted

from datetime import datetime
#from time import time
import time
from numbers import Number
from copy import deepcopy
from warnings import warn, simplefilter, catch_warnings
from textwrap import dedent

import numpy as np
from tqdm import tqdm
import pickle as pkl 

import multiprocessing
from joblib import Parallel, delayed
#from mvmm.utils import get_seeds
#from mvmm.opt_utils import check_stopping_criteria

def check_stopping_criteria(abs_diff=None, rel_diff=None, abs_tol=None, rel_tol=None):
    """
    Decides whether or not to stop an optimization algorithm if the relative and/or absolute difference stopping critera are met. If both abs_tol and rel_tol are not None, then will only stop if both conditions are met.

    Parameters
    ----------
    abs_diff: float, None
        The absolute difference in succesive loss functions.

    rel_diff: float, None
        The relative difference in succesive loss functions.

    abs_tol: None, float
        The absolute difference tolerance.

    rel_tol: None, float
        The relative difference tolerance.

    Output
    ------
    stop: bool
    """
    
    if abs_tol is not None and abs_diff is not None and abs_diff <= abs_tol:
        a_stop = True
    else:
        a_stop = False
    
    if rel_tol is not None and rel_diff is not None and rel_diff <= rel_tol:
        r_stop = True
    else:
        r_stop = False
    
    if abs_tol is not None and rel_tol is not None:
        # if both critera are use both must be true
        return a_stop and r_stop
    else:
        # otherwise stop if either is True
        return a_stop or r_stop

class MixtureModelMixin(metaclass=ABCMeta):
    """
    Base mixture model class.
    """
    def _set_nan_mask(self, X):
        self._nan_mask = np.isnan(X)
    
    @abstractmethod
    def _get_parameters(self):
        raise NotImplementedError
    
    @abstractmethod
    def _set_parameters(self, params):
        raise NotImplementedError
    
    @abstractmethod
    def _get_resps(self):
        raise NotImplementedError
    
    @abstractmethod
    def _set_resps(self, params):
        raise NotImplementedError
    
    def fit_vem(self, X):
        """
        Fits the mixture model.

        Parameters
        ----------
        X:
            The observed data.
        """
        X = _check_X(X, self.n_row_components, self.n_col_components, ensure_min_samples=2)
        self.metadata_ = {'n_samples': X.shape[0],
                          'n_features': X.shape[1]}
        self._check_parameters(X)
        self._check_fitting_parameters(X)

        start_time = time()
        self._fit_vem(X)
        self.metadata_['fit_time'] = time() - start_time
        return self
    
    def fit_sem(self, X):
        """
        Fits the mixture model.

        Parameters
        ----------
        X:
            The observed data.
        """
        X = _check_X(X, self.n_row_components, self.n_col_components, ensure_min_samples=2)
        self.metadata_ = {'n_samples': X.shape[0],
                          'n_features': X.shape[1]}
        self._check_parameters(X)
        self._check_fitting_parameters(X)

        start_time = time()
        self._fit_sem(X)
        self.metadata_['fit_time'] = time() - start_time
        return self

    @abstractmethod
    def _fit_vem(self, X):
        # subclass should implement this!
        raise NotImplementedError

    @abstractmethod
    def _fit_sem(self, X):
        # subclass should implement this!
        raise NotImplementedError

    def _check_parameters(self, X):
        """
        Check values of the basic parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        """
        # sub-class should overwrite
        pass

    def _check_clust_param_values(self, X):
        """Check values of the basic fitting parameters.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        """
        pass

    def _check_clust_parameters(self, X):
        """
        Checks cluster parameters and weights.
        """
        pass

    def score_samples(self, X):
        """
        Computes the observed data log-likelihood for each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        log_prob : array, shape (n_samples,)
            Log probabilities of each data point in X.
        """
        raise NotImplementedError

    def row_log_probs(self, X):
        """
        Computes the log-likelihood for each sample for each cluster including the cluster weihts.

        Parameters
        ----------
        X: array-like, (n_samples, n_features)

        Output
        ------
        log_probs: array-like, (n_samples, n_components)
        """
        #check_is_fitted(self)
        # formerly _estimate_weighted_log_prob
        return self.comp_row_log_probs(X) + np.log(self.row_weights_)

    def col_log_probs(self, X, v):
        """
        Computes the log-likelihood for each sample for each cluster including the cluster weihts.

        Parameters
        ----------
        X: array-like, (n_samples, n_features)

        Output
        ------
        log_probs: array-like, (n_samples, n_components)
        """
        #check_is_fitted(self)
        # formerly _estimate_weighted_log_prob
        return self.comp_col_log_probs(X, v) + np.log(self.view_models_[v].col_weights_)

    @abstractmethod
    def comp_row_log_probs(self, X):
        raise NotImplementedError
    
    @abstractmethod
    def comp_col_log_probs(self, X, v):
        raise NotImplementedError
    
    def score(self, X, y=None):
        """Compute the per-sample average log-likelihood of the given data X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_dimensions)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        Returns
        -------
        log_likelihood : float
            Log likelihood of the fussian mixture given X.
        """
        raise NotImplementedError

    def log_likelihood(self, X):
        """
        Computes the observed data log-likelihood.

        """
        #check_is_fitted(self)
        return self.score_samples(X).sum()

    def predict(self, X):
        pass
    
    def predict_proba(self, X):
        pass
    
    def log_resps(self, log_prob):
        """
        Estimate log probabilities and responsibilities for each sample.
        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : array, shape (n_samples,)
            log p(X)
        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities
        """
        # weighted_log_prob = self.log_probs(X)
        log_prob_norm = logsumexp(log_prob, axis=1)
        with np.errstate(under='ignore'):
            # ignore underflow
            log_resps = log_prob - log_prob_norm[:, np.newaxis]
        return log_resps

    def sample(self, n_samples=1, random_state=None):
        """
        Generate random samples from the model.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Defaults to 1.

        random_state: int, None
            Random seed.

        Returns
        -------
        X : array-like, shape (n_samples, n_features)
            List of samples

        """
        #check_is_fitted(self)

        rng = check_random_state(random_state)
        pi = self.weights_
        y = rng.choice(a=np.arange(len(pi)), size=n_samples,
                       replace=True, p=pi)

        samples = [None for _ in range(n_samples)]
        for i in range(n_samples):
            samples[i] = self.sample_from_comp(y=y[i], random_state=rng)

        return np.array(samples), y

    @abstractmethod
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
        raise NotImplementedError

    def bic(self, X):
        """
        Bayesian information criterion for the current model fit
        and the proposed data

        Parameters
        ----------
        X : array of shape(n_samples, n_featuers)

        Returns
        -------
        bic: float (the lower the better)
        """
        #check_is_fitted(self)
        n = X.shape[0]
        return -2 * self.log_likelihood(X) + np.log(n) * self._n_parameters()

    def aic(self, X):
        """
        Akaike information criterion for the current model fit
        and the proposed data.

        Parameters
        ----------
        X : array of shape(n_samples, n_featuers)
        Returns
        -------
        aic: float (the lower the better)
        """
        #check_is_fitted(self)
        return -2 * self.log_likelihood(X) + 2 * self._n_parameters()
    
    def computeICL(self, X):
        raise NotImplementedError
    
    def _n_parameters(self):
        """
        Returns the number of model parameters e.g. for BIC/AIC.
        """
        #check_is_fitted(self)
        return self._n_cluster_parameters() + self._n_weight_parameters()

    def _n_weight_parameters(self):
        """
        Number of weight parameters
        """
        return self.n_components - 1

    @abstractmethod
    def _n_cluster_parameters(self):
        raise NotImplementedError

    def reorder_components(self, new_idxs):
        """
        Re-orders the components
        """
        assert set(new_idxs) == set(range(self.n_components))
        params = self._reorder_component_params(new_idxs)
        params['weights'] = self.weights_[new_idxs]
        self._set_parameters(params)
        return self
    
    def reduce_n_row_components(self):
        self.n_row_components = self.n_row_components -1 
    
    def drop_component(self, comps):
        """
        Drops a component or components from the model.

        Parameters
        ----------
        comps: int, list of ints
            Which component(s) to drop
        """
        #check_is_fitted(self)
        if isinstance(comps, Number):
            comps = [comps]

        # sort componets in decreasing order so that lower indicies
        # are preserved after dropping higher indices
        comps = np.sort(comps)[::-1]

        # don't drop every component
        assert len(comps) < self.n_row_components

        row_weights = deepcopy(self.row_weights_)
        row_resps = deepcopy(self.row_resps_)
        
        for k in comps:
            
            self.reduce_n_row_components()
            params = self._drop_component_params(k)
            row_weights = np.delete(row_weights, k)
            params['row_weights'] = row_weights / sum(row_weights)
            self._set_parameters(params)
            
            row_resps = np.delete(row_resps, k, 1)
            row_resps = row_resps / np.sum(row_resps, 1)[:, np.newaxis]
            self._set_resps({'row_resps': row_resps})

        return self

    def _drop_component_params(self, k):
        """
        Drops the cluster parameters of a single componet.
        Subclass should overwrite.
        """
        raise NotImplementedError

    def _reorder_component_params(self, new_idxs):
        """
        Re-orders the component cluster parameters
        """
        raise NotImplementedError


class EMfitMMMixin(metaclass=ABCMeta):
    """
    Based EM mixture model class.
    """

    def __init__(self,
                 max_n_steps=200,
                 burn_in_steps=100,
                 resampling_steps=100,
                 resampling_fraction=0.2,
                 row_max_n_steps=1,
                 col_max_n_steps=1,
                 abs_tol=1e-9,
                 rel_tol=None,
                 n_init=5,
                 init_method='rand',
                 init_input_method=None,
                 init_resps=None,
                 init_params=None,
                 #init_params_method='rand_resps',
                 #init_params_value=None,
                 #init_resps_method='rand_resps',
                 #init_resps_value=None,
                 #init_row_weights_method="uniform",
                 #init_row_weights_value=None,
                 random_state=None,
                 verbosity=0,
                 history_tracking=0):

        self.max_n_steps = max_n_steps
        self.burn_in_steps = burn_in_steps
        self.resampling_steps = resampling_steps
        self.resampling_fraction = resampling_fraction
        self.row_max_n_steps = row_max_n_steps
        self.col_max_n_steps = col_max_n_steps
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.n_init = n_init
        
        self.init_method=init_method
        self.init_input_method=init_input_method
        self.init_resps=init_resps
        self.init_params=init_params
        
        #self.init_params_method = init_params_method
        #self.init_params_value = init_params_value
        #self.init_resps_method = init_resps_method
        #self.init_resps_value = init_resps_value
        #self.init_row_weights_method=init_row_weights_method
        #self.init_row_weights_value=init_row_weights_value
        self.random_state = random_state

        self.verbosity = verbosity
        self.history_tracking = history_tracking

    def _fit_vem(self, X):
        params, self.opt_data_ = self._best_vem_loop(X)
        self._set_parameters(params)

    def _fit_sem(self, X):
        params, self.opt_data_ = self._best_sem_loop(X)
        self._set_parameters(params)

    def _check_fitting_parameters(self, X):

        if self.n_row_components < 1:
            raise ValueError("Invalid value for 'n_row_components': %d "
                             "Estimation requires at least one component"
                             % self.n_row_components)
        
        if  self.n_row_components > X.shape[0]:
            raise ValueError("Invalid value for 'n_row_components': %d "
                             "Cannot be greater than the number of samples"
                             % self.n_row_components)
        
        if self.n_col_components < 1:
            raise ValueError("Invalid value for 'n_col_components': %d "
                             "Estimation requires at least one component"
                             % self.n_col_components)
        
        if  self.n_col_components > X.shape[1]:
            raise ValueError("Invalid value for 'n_col_components': %d "
                             "Cannot be greater than the number of features"
                             % self.n_col_components)
        
        if self.abs_tol is not None and self.abs_tol < 0.:
            raise ValueError("Invalid value for 'abs_tol': %.5f "
                             "Tolerance must be non-negative"
                             % self.abs_tol)

        if self.rel_tol is not None and self.rel_tol < 0.:
            raise ValueError("Invalid value for 'rel_tol': %.5f "
                             "Tolerance must be non-negative"
                             % self.rel_tol)

        if self.n_init < 1:
            raise ValueError("Invalid value for 'n_init': %d "
                             "Estimation requires at least one run"
                             % self.n_init)

        if self.max_n_steps < 0:
            raise ValueError("Invalid value for 'max_n_steps': %d "
                             ", must be non negative."
                             % self.max_n_steps)
    
    def _sampling_imputation_initialization(self, X, init_resps, random_state):
        row_idx, col_idx = np.where(self._nan_mask)
        if len(row_idx) > 0:
          row_components = np.argmax(init_resps['row_resps'], 1)
          col_components = np.argmax(init_resps['col_resps'], 1)
          row_component = row_components[row_idx]
          col_component = col_components[col_idx]
          comp_pairs = np.stack(list(zip(row_component, col_component)))
          for comp_pair in np.unique(comp_pairs, axis = 0):
            idx = np.where((comp_pairs == comp_pair).all(axis = 1))[0]
            row_c = comp_pair[0]
            col_c = comp_pair[1]
            row_comp_idx = np.where(row_components == row_c)[0]
            col_comp_idx = np.where(col_components == col_c)[0]
            values = X[row_comp_idx, :][:, col_comp_idx].flatten()
            values = values[~np.isnan(values)]
            X[row_idx[idx], col_idx[idx]] = random_state.choice(values, size = len(idx))
        
        return X
    
    def initialize_parameters(self, X, random_state=None):
        random_state = check_random_state(random_state)
        init_resps = self.get_init_resps(X, random_state)
        init_row_weights, init_col_weights = self._get_init_weights(X, init_resps, random_state)
        X = self._sampling_imputation_initialization(X, init_resps, random_state)
        init_params = self._get_init_clust_parameters(X, init_resps, random_state)
        init_params['row_weights'] = init_row_weights
        init_params['col_weights'] = init_col_weights
        # over write initialized parameters with user provided parameters
        init_params = self._update_user_init(init_params=init_params)
        
        init_params = self._zero_initialization(init_params=init_params)
        self._set_parameters(params=init_params)
        
        self._set_resps(resps=init_resps)
        
        self._check_clust_parameters(X)
    
    def _zero_initialization(self, init_params):
        """
        Called after initialization; subclass may optinoally overwrite.

        Output
        ------
        dict that gets pass to _set_parameters()
        """
        return init_params

    def get_init_resps(self, X, random_state):
        #if self.init_resps_method == 'rand_resps':
        n_samples, n_features = X.shape
        row_assign = np.arange(self.n_row_components)
        row_assign = np.append(row_assign, random_state.choice(range(self.n_row_components), size = n_samples - self.n_row_components))
        random_state.shuffle(row_assign)
        row_resps = np.zeros((row_assign.size, row_assign.max() + 1))
        row_resps[np.arange(row_assign.size), row_assign] = 1
        print(row_resps.sum(0))
        col_assign = np.arange(self.n_col_components)
        col_assign = np.append(col_assign, random_state.choice(range(self.n_col_components), size = n_features - self.n_col_components))
        random_state.shuffle(col_assign)
        col_resps = np.zeros((col_assign.size, col_assign.max() + 1))
        col_resps[np.arange(col_assign.size), col_assign] = 1
        print(col_resps.sum(0))
        #row_resps = random_state.rand(n_samples, self.n_row_components)
        #row_resps /= row_resps.sum(axis=1)[:, np.newaxis]
        #col_resps = random_state.rand(n_features, self.n_col_components)
        #col_resps /= col_resps.sum(axis=1)[:, np.newaxis]
        
        return {'row_resps': row_resps, 'col_resps': col_resps}
        
    def _get_init_clust_parameters(self, X, init_resps, random_state):

        init_params = self._sm_step_clust_params(X, init_resps['row_resps'], init_resps['col_resps'])

        return init_params

    def _get_init_weights(self, X, init_resps, random_state):
        #Requries changing the method for some reason, ('uniform', )
        #if self.init_col_weights_method == 'uniform':
        init_row_weights = init_resps['row_resps'].sum(0) / init_resps['row_resps'].sum()
        
        init_col_weights = init_resps['col_resps'].sum(0) / init_resps['col_resps'].sum()

        #else:
        #    raise ValueError("Invalid value for 'init_col_weights_method': {}"
                   #          "".format(self.init_col_weights_method))

        return init_row_weights, init_col_weights

    def _resample_null_row_components(self, resps, random_state):
          return self._resample_null_components(resps, random_state)
    
    def _resample_null_col_components(self, resps, random_state):
          return self._resample_null_components(resps, random_state)

    def _resample_null_components(self, resps, random_state):
          random_state = check_random_state(random_state)
          n_samples, n_components = resps.shape
          n_resample = np.round(n_samples * self.resampling_fraction).astype(int)
          #empty = np.where(resps.sum(0)==0)[0]
          #n_resample2 = len(empty)
          #n_resample = np.max(n_resample1, n_resample2)
          resampling_index = random_state.choice(range(n_samples), size = n_resample, replace = False)
          if n_resample < n_components:
            resample_values = random_state.choice(range(n_components), size = n_resample)
          else:
            resample_values = np.append(np.arange(n_components), random_state.choice(range(n_components), size = n_resample - n_components))
            random_state.shuffle(resample_values)
          
          #resample_values = random_state.choice(range(n_components), size = n_resample)
          resampled_resps = np.zeros((n_resample,n_components))
          resampled_resps[np.arange(resample_values.size), resample_values] = 1
          resps[resampling_index, :] = resampled_resps
          return resps
    
    def _update_user_init(self, init_params=None):

        if init_params is None:
            init_params = {}

        # drop parameters whom the user has provided initial values for
        if self.init_params is not None:
            for k in self.init_params:
                if k in init_params.keys():
                    init_params[k] = self.init_params[k]

        return init_params
    
    def _combine_SEM_params(self, params_list):
        pass
    
    def _obs_nll(self, X):
        
        obs_nll = -np.sum(self.row_resps_ * np.log(self.row_resps_)) + self.row_resps_.sum(0) @ np.log(self.row_weights_)
        
        for v in range(self.n_views):
            obs_nll += self.view_models_[v]._get_obs_nll(X[v])
        
        return obs_nll

    def _row_m_step(self, X, E_out):
        resps = E_out['resps']

        new_params = self._row_m_step_clust_params(X=X, resps=resps)
        new_params['row_weights'] = self._row_m_step_weights(X=X, resps=resps)

        return new_params
    
    def _col_m_step(self, X, E_out, v):
        resps = E_out['resps']

        new_params = self._col_m_step_clust_params(X=X, resps=resps, v=v)
        new_params['col_weights'] = self._col_m_step_weights(X=X, resps=resps)

        return new_params

    def _row_sm_step(self, X, E_out):
        resps = E_out['resps']

        new_params = self._row_sm_step_clust_params(X=X, resps=resps)
        new_params['row_weights'] = self._row_sm_step_weights(X=X, resps=resps)

        return new_params
    
    def _col_sm_step(self, X, E_out, v):
        resps = E_out['resps']

        new_params = self._col_sm_step_clust_params(X=X, resps=resps, v=v)
        new_params['col_weights'] = self._col_sm_step_weights(X=X, resps=resps)

        return new_params

    def _row_m_step_weights(self, X, resps):
        n_samples = resps.shape[0]
        nk = resps.sum(axis=0) + 10 * np.finfo(resps.dtype).eps
        return nk / n_samples

    def _col_m_step_weights(self, X, resps):
        n_features = resps.shape[0]
        nk = resps.sum(axis=0) + 10 * np.finfo(resps.dtype).eps
        return nk / n_features
    
    def _row_sm_step_weights(self, X, resps):
        n_samples = resps.shape[0]
        nk = resps.sum(axis=0) + 10 * np.finfo(resps.dtype).eps
        return nk / n_samples

    def _col_sm_step_weights(self, X, resps):
        n_features = resps.shape[0]
        nk = resps.sum(axis=0) + 10 * np.finfo(resps.dtype).eps
        return nk / n_features

    def _m_step_clust_params(self, X, resps):
        raise NotImplementedError

    def _sm_step_clust_params(self, X, resps):
        raise NotImplementedError

    def _row_ve_step(self, X, random_state):
        """

        Parameters
        ----------
        X: array-like, (n_samples, n_features)
            The data.

        Output
        ------
        out: dict

        out['log_resp']: array-like, (n_samples, n_components)
            The responsitiblities.

        out['obs_nll']: float
            The observed negative log-likelihood of the data at the current
            parameters.
        """
        log_prob = self.row_log_probs(X)
        log_resps = self.log_resps(log_prob)
        
        #self.row_resps_ = np.exp(log_resp)
        
        obs_nll = - self._obs_nll(X)

        return {'resps': np.exp(log_resps), 'obs_nll': obs_nll}

    def _col_ve_step(self, X, v, random_state):
        """

        Parameters
        ----------
        X: array-like, (n_samples, n_features)
            The data.

        Output
        ------
        out: dict

        out['log_resps']: array-like, (n_samples, n_components)
            The respsonsitiblities.

        out['obs_nll']: float
            The observed negative log-likelihood of the data at the current
            parameters.
        """
        log_prob = self.col_log_probs(X, v)
        log_resps = self.log_resps(log_prob)
        #self.view_models_[v].col_resps_ = np.exp(log_resps)

        return {'resps': np.exp(log_resps)}

    def _row_se_step(self, X, random_state):
        """

        Parameters
        ----------
        X: array-like, (n_samples, n_features)
            The data.

        Output
        ------
        out: dict

        out['log_resp']: array-like, (n_samples, n_components)
            The responsitiblities.

        out['obs_nll']: float
            The observed negative log-likelihood of the data at the current
            parameters.
        """
        random_state = check_random_state(random_state)
        log_prob = self.row_log_probs(X)
        gumbel_samp = log_prob + random_state.gumbel(0, size = log_prob.shape)
        #prob = np.exp(log_prob)
        #prob = prob / prob.sum(1)[:, np.newaxis]
        #row_assign = (prob.cumsum(1) > random_state.rand(prob.shape[0])[:,None]).argmax(1)
        row_assign = gumbel_samp.argmax(1)
        resps = np.zeros((row_assign.size, self.n_row_components))
        resps[np.arange(row_assign.size), row_assign] = 1
        #log_resps = self.log_resps(log_prob)
        
        #self.row_resps_ = np.exp(log_resp)
        
        #obs_nll = - self._obs_nll(X)
        return {'resps': resps}
        #return {'resps': resps, 'obs_nll': obs_nll}

    def _col_se_step(self, X, v, random_state):
        """

        Parameters
        ----------
        X: array-like, (n_samples, n_features)
            The data.

        Output
        ------
        out: dict

        out['log_resps']: array-like, (n_samples, n_components)
            The respsonsitiblities.

        out['obs_nll']: float
            The observed negative log-likelihood of the data at the current
            parameters.
        """
        random_state = check_random_state(random_state)
        log_prob = self.col_log_probs(X, v)
        self.log_prob = log_prob
        _, n_col_components = log_prob.shape
        gumbel_samp = log_prob + random_state.gumbel(0, size = log_prob.shape)
        col_assign = gumbel_samp.argmax(1)
        #prob = np.exp(log_prob)
        #prob = prob / prob.sum(1)[:, np.newaxis]
        #col_assign = (prob.cumsum(1) > random_state.rand(prob.shape[0])[:,None]).argmax(1)
        resps = np.zeros((col_assign.size, n_col_components))
        resps[np.arange(col_assign.size), col_assign] = 1
        #log_resps = self.log_resps(log_prob)
        
        #log_prob = self.col_log_probs(X, v)
        #log_resps = self.log_resps(log_prob)
        #self.view_models_[v].col_resps_ = np.exp(log_resps)

        return {'resps': resps}

    def compute_tracking_data(self, X, E_out=None):
        """
        Parameters
        ----------
        X: array-like, (n_samples, n_features)
            The data.

        E_out: None, dict
            (optional) The output from _e_step which includes the
            observe neg log lik. Saves computational time.

        Output
        ------
        dict:
            out['obs_nll']: float
                The observed neg log-lik.

            out['loss_val']: float
                The loss function; in this case just obs_nll.

            out['model']: (only if history_tracking >=2)
                The current model parameters.
        """
        out = {}

        #if E_out is not None:
            #out['obs_nll'] = E_out['obs_nll']
            #out['icl'] = E_out['icl']
        #else:
            #out['obs_nll'] = - self.score(X)
            #out['icl'] = self.computeICL(X)
        
        
        # obs_nll = - self.score(X)
        # log_probs = self.log_probs(X)
        # obs_nll = - logsumexp(log_probs, axis=1).mean()
        #out['loss_val'] = out['obs_nll']
        
        out['model'] = deepcopy(self._get_parameters())

        return out
    
    def _row_vem_loop(self, X, resampling_step, random_state):
        
        # make sure the cluster parameters have been properly initialized
        self._check_clust_param_values(X)

        # initialize history tracking
        history = {}

        converged = False
        prev_loss = None

        start_time = time.time()

        current_loss = np.nan
        converged = False
        step = -1

        if self.history_tracking >= 1:
            history['init_params'] = self._get_parameters()

        #for step in range(self.row_max_n_steps):
        if self.verbosity >= 2:
            t = datetime.now().strftime("%H:%M:%S")
            print('EM step {} at {}'.format(step + 1, t))

        ################################
        # E-step and check convergence #
        ################################
        E_out = self._row_ve_step(X = X, random_state = random_state)
        new_resps = {}
        new_resps['row_resps'] = E_out['resps']
        while resampling_step and any(new_resps['row_resps'].sum(0) == 0):
          new_resps['row_resps'] = self._resample_null_row_components(new_resps['row_resps'], random_state)
        
        new_resps['row_resps'].shape
        
        #Set the new responsiblilities
        self._set_resps(new_resps)
        
        #E_out['icl'] = self.computeICL(X)
        tracking_data = self.compute_tracking_data(X, E_out)
        #current_loss = tracking_data['loss_val']
        #current_icl = tracking_data['icl']
        
        # track data
        for k in tracking_data.keys():
            if k not in history.keys():
                history[k] = []
            history[k].append(tracking_data[k])
        
        new_params = self._row_m_step(X=X, E_out=E_out)
        self._set_parameters(new_params)

        prev_loss = deepcopy(current_loss)

        params = deepcopy(self._get_parameters())

        opt_data = {#'loss_val': current_loss,
                    #'icl': current_icl,
                    'n_steps': step,
                    'runtime': time.time() - start_time,
                    'success': True,  # subclasses may have failed EM loop
                    'history': history}

        if not converged and self.verbosity >= 1:
            warn('EM did not converge', ConvergenceWarning)

        return params, opt_data, E_out
    
    def _col_vem_loop(self, X, resampling_step, random_state):
        
        new_view_params = []
        new_view_resps = []
        
        for v in range(self.n_views):
            # make sure the cluster parameters have been properly initialized
            self._check_clust_param_values(X)
    
            # initialize history tracking
            history = {}
    
            converged = False
            prev_loss = None
    
            start_time = time.time()
    
            current_loss = np.nan
            converged = False
            step = -1
    
            if self.history_tracking >= 1:
                history['init_params'] = self._get_parameters()
        
            if self.verbosity >= 2:
                t = datetime.now().strftime("%H:%M:%S")
                print('EM step {} at {}'.format(step + 1, t))

            ################################
            # E-step and check convergence #
            ################################
            E_out = self._col_ve_step(X[v], v, random_state)
            #E_out['obs_nll'] = - self._obs_nll(X)
            new_resps = {}
            new_resps['col_resps'] = E_out['resps']
            
            l = 0 
            while resampling_step and any(new_resps['col_resps'].sum(0) == 0):
                resampled_resps = self.view_models_[v]._resample_null_col_components(new_resps['col_resps'], random_state + l)
                new_resps['col_resps'] = resampled_resps
                l += 1
            
            new_view_resps.append(new_resps)
            new_params = self._col_m_step(X=X[v], E_out=E_out, v=v)
            # update new parameters
            new_view_params.append(new_params)

            prev_loss = deepcopy(current_loss)
        
        new_params = {'views': new_view_params}
        new_resps = {'views': new_view_resps}
        self._set_parameters(new_params)
        self._set_resps(new_resps)
        #E_out['icl'] = self.computeICL(X)
        #current_loss = tracking_data['loss_val']
        tracking_data = self.compute_tracking_data(X, E_out)
        #Set the new responsiblilities
        for k in tracking_data.keys():
            if k not in history.keys():
                history[k] = []
            history[k].append(tracking_data[k])
        
        #current_icl = tracking_data['icl']
        # track data
        params = deepcopy(self._get_parameters())

        opt_data = {#'loss_val': current_loss,
                    #'icl': current_icl,
                    'n_steps': step,
                    'runtime': time.time() - start_time,
                    'success': True,  # subclasses may have failed EM loop
                    'history': history}

        if not converged and self.verbosity >= 1:
            warn('EM did not converge', ConvergenceWarning)

        return params, opt_data, E_out
    
    def _row_sem_loop(self, X, resampling_step, random_state):
        
        # make sure the cluster parameters have been properly initialized
        self._check_clust_param_values(X)

        # initialize history tracking
        history = {}

        converged = False
        prev_loss = None

        start_time = time.time()

        current_loss = np.nan
        converged = False
        step = -1

        if self.history_tracking >= 1:
            history['init_params'] = self._get_parameters()

        #for step in range(self.row_max_n_steps):
        if self.verbosity >= 2:
            t = datetime.now().strftime("%H:%M:%S")
            print('EM step {} at {}'.format(step + 1, t))

        ################################
        # E-step and check convergence #
        ################################
        E_out = self._row_se_step(X = X, random_state = random_state)
        new_resps = {}
        new_resps['row_resps'] = E_out['resps']
        while resampling_step and any(new_resps['row_resps'].sum(0) == 0):
          new_resps['row_resps'] = self._resample_null_row_components(new_resps['row_resps'], random_state)
        
        if any(new_resps['row_resps'].sum(0) == 0):
          raise ValueError        
        
        #Set the new responsiblilities
        self._set_resps(new_resps)
        
        #E_out['icl'] = self.computeICL(X)
        tracking_data = self.compute_tracking_data(X, E_out)
        #current_loss = tracking_data['loss_val']
        #current_icl = tracking_data['icl']
        
        # track data
        for k in tracking_data.keys():
            if k not in history.keys():
                history[k] = []
            history[k].append(tracking_data[k])
        
        new_params = self._row_sm_step(X=X, E_out=E_out)
        self._set_parameters(new_params)

        prev_loss = deepcopy(current_loss)

        params = deepcopy(self._get_parameters())

        opt_data = {#'loss_val': current_loss,
                    #'icl': current_icl,
                    'n_steps': step,
                    'runtime': time.time() - start_time,
                    'success': True,  # subclasses may have failed EM loop
                    'history': history}

        if not converged and self.verbosity >= 1:
            warn('EM did not converge', ConvergenceWarning)

        return params, opt_data, E_out
    
    def _col_sem_loop(self, X, resampling_step, random_state):
        
        new_view_params = []
        new_view_resps = []
        
        for v in range(self.n_views):
            # make sure the cluster parameters have been properly initialized
            self._check_clust_param_values(X)
    
            # initialize history tracking
            history = {}
    
            converged = False
            prev_loss = None
    
            start_time = time.time()
    
            current_loss = np.nan
            converged = False
            step = -1
    
            if self.history_tracking >= 1:
                history['init_params'] = self._get_parameters()
        
            if self.verbosity >= 2:
                t = datetime.now().strftime("%H:%M:%S")
                print('EM step {} at {}'.format(step + 1, t))

            ################################
            # E-step and check convergence #
            ################################
            E_out = self._col_se_step(X[v], v, random_state)
            #E_out['obs_nll'] = - self._obs_nll(X)
            new_resps = {}
            new_resps['col_resps'] = E_out['resps']
            
            l = 0 
            while resampling_step and any(new_resps['col_resps'].sum(0) == 0):
                resampled_resps = self.view_models_[v]._resample_null_col_components(new_resps['col_resps'], random_state + l)
                new_resps['col_resps'] = resampled_resps
                l += 1
            
            if any(new_resps['col_resps'].sum(0) == 0):
              raise ValueError
            
            new_view_resps.append(new_resps)
            new_params = self._col_sm_step(X=X[v], E_out=E_out, v=v)
            # update new parameters
            new_view_params.append(new_params)

            prev_loss = deepcopy(current_loss)
        
        new_params = {'views': new_view_params}
        new_resps = {'views': new_view_resps}
        self._set_parameters(new_params)
        self._set_resps(new_resps)
        #E_out['icl'] = self.computeICL(X)
        tracking_data = self.compute_tracking_data(X, E_out)
        #current_loss = tracking_data['loss_val']
        tracking_data = self.compute_tracking_data(X, E_out)
        #Set the new responsiblilities
        for k in tracking_data.keys():
            if k not in history.keys():
                history[k] = []
            history[k].append(tracking_data[k])
        
        #current_icl = tracking_data['icl']
        # track data
        params = deepcopy(self._get_parameters())

        opt_data = {#'loss_val': current_loss,
                    #'icl': current_icl,
                    'n_steps': step,
                    'runtime': time.time() - start_time,
                    'success': True,  # subclasses may have failed EM loop
                    'history': history}

        if not converged and self.verbosity >= 1:
            warn('EM did not converge', ConvergenceWarning)

        return params, opt_data, E_out
    
    def _impute_missing_data(self, X, random_state):
        print("Running Pass")
        pass
    
    def _sem_loop(self, X, random_state):

        # make sure the cluster parameters have been properly initialized
        self._check_clust_param_values(X)

        # initialize history tracking
        history = {}

        converged = False
        prev_loss = None 

        start_time = time.time()
        self.params_his = []
        current_loss = np.nan
        converged = False
        step = -1
        
        
        if self.history_tracking >= 1:
            history['init_params'] = self._get_parameters()

        for step in tqdm(range(self.max_n_steps)):
            
            if step < self.resampling_steps:
                resampling_step = True
            else:
                resampling_step = False
            
            #print("Step: ",step)
            if self.verbosity >= 2:
                t = datetime.now().strftime("%H:%M:%S")
                print('EM step {} at {}'.format(step + 1, t))
            
            # Row EM loop
            with catch_warnings():
                simplefilter('ignore', ConvergenceWarning)
                params, opt_data, E_out = self._row_sem_loop(X=X, resampling_step=resampling_step, random_state=random_state)
            
            tracking_data = self.compute_tracking_data(X, E_out)
            #current_loss = tracking_data['loss_val']
            #current_icl = tracking_data['icl']
            # track data
            for k in tracking_data.keys():
                if k not in history.keys():
                    history[k] = []
                history[k].append(tracking_data[k])

            prev_loss = deepcopy(current_loss)
            
            # Col EM loop
            with catch_warnings():
                simplefilter('ignore', ConvergenceWarning)
                params, opt_data, E_out = self._col_sem_loop(X=X, resampling_step = resampling_step, random_state =random_state)
            #
            tracking_data = self.compute_tracking_data(X, E_out)
            #current_loss = tracking_data['loss_val']
            #current_icl = tracking_data['icl']
            # track data
            for k in tracking_data.keys():
                if k not in history.keys():
                    history[k] = []
                history[k].append(tracking_data[k])
            
            prev_loss = deepcopy(current_loss)
            X = self._impute_missing_data(X=X, random_state= random_state)
            if 'params' not in history.keys():
              history['params'] = [params]
              self.params_his = [params]
            else:
              history['params'].append(params)
              self.params_his.append(params)
        
        params = self._combine_SEM_params([self.params_his[i] for i in range(self.burn_in_steps, self.max_n_steps)])
        self.params_his.append(params)
        self._set_parameters(params)
        with catch_warnings():
            simplefilter('ignore', ConvergenceWarning)
            params, opt_data, E_out = self._row_sem_loop(X=X, resampling_step=resampling_step, random_state=random_state)
        
        tracking_data = self.compute_tracking_data(X, E_out)
        #current_loss = tracking_data['loss_val']
        #current_icl = tracking_data['icl']
        
        # track data
        for k in tracking_data.keys():
            if k not in history.keys():
                history[k] = []
            history[k].append(tracking_data[k])

        prev_loss = deepcopy(current_loss)
        
        # Col EM loop
        with catch_warnings():
            simplefilter('ignore', ConvergenceWarning)
            params, opt_data, E_out = self._col_sem_loop(X=X, resampling_step = resampling_step, random_state =random_state)
        #
        tracking_data = self.compute_tracking_data(X, E_out)
        #current_loss = tracking_data['loss_val']
        #current_icl = tracking_data['icl']
        
        # track data
        for k in tracking_data.keys():
            if k not in history.keys():
                history[k] = []
            history[k].append(tracking_data[k])
        
        prev_loss = deepcopy(current_loss)
        
        X = self._impute_missing_data(X=X, random_state= random_state)
        
        params = self._get_parameters()

        opt_data = {#'loss_val': current_loss,
                    'icl': self.computeICL(X),
                    'n_steps': step,
                    'runtime': time.time() - start_time,
                    'success': True,  # subclasses may have failed EM loop
                    'history': history}

        return params, opt_data

    def _vem_loop(self, X, random_state):

        # make sure the cluster parameters have been properly initialized
        self._check_clust_param_values(X)

        # initialize history tracking
        history = {}

        converged = False
        prev_loss = None 

        start_time = time.time()
        self.params_his = []
        current_loss = np.nan
        converged = False
        step = -1
        
        
        if self.history_tracking >= 1:
            history['init_params'] = self._get_parameters()

        for step in tqdm(range(self.max_n_steps)):
            
            if step < self.resampling_steps:
                resampling_step = True
            else:
                resampling_step = False
            
            #print("Step: ",step)
            if self.verbosity >= 2:
                t = datetime.now().strftime("%H:%M:%S")
                print('EM step {} at {}'.format(step + 1, t))
            
            # Row EM loop
            with catch_warnings():
                simplefilter('ignore', ConvergenceWarning)
                params, opt_data, E_out = self._row_vem_loop(X=X, resampling_step=resampling_step, random_state=random_state)
            
            tracking_data = self.compute_tracking_data(X, E_out)
            #current_loss = tracking_data['loss_val']
            #current_icl = tracking_data['icl']
            # track data
            for k in tracking_data.keys():
                if k not in history.keys():
                    history[k] = []
                history[k].append(tracking_data[k])

            prev_loss = deepcopy(current_loss)
            
            # Col EM loop
            with catch_warnings():
                simplefilter('ignore', ConvergenceWarning)
                params, opt_data, E_out = self._col_vem_loop(X=X, resampling_step = resampling_step, random_state =random_state)
            #
            tracking_data = self.compute_tracking_data(X, E_out)
            #current_loss = tracking_data['loss_val']
            #current_icl = tracking_data['icl']
            # track data
            for k in tracking_data.keys():
                if k not in history.keys():
                    history[k] = []
                history[k].append(tracking_data[k])
            
            prev_loss = deepcopy(current_loss)
            X = self._impute_missing_data(X=X, random_state= random_state)
            if 'params' not in history.keys():
              history['params'] = [params]
              self.params_his = [params]
            else:
              history['params'].append(params)
              self.params_his.append(params)
        
        self.params_his.append(params)
        self._set_parameters(params)
        with catch_warnings():
            simplefilter('ignore', ConvergenceWarning)
            params, opt_data, E_out = self._row_vem_loop(X=X, resampling_step=resampling_step, random_state=random_state)
        
        tracking_data = self.compute_tracking_data(X, E_out)
        #current_loss = tracking_data['loss_val']
        #current_icl = tracking_data['icl']
        
        # track data
        for k in tracking_data.keys():
            if k not in history.keys():
                history[k] = []
            history[k].append(tracking_data[k])

        prev_loss = deepcopy(current_loss)
        
        # Col EM loop
        with catch_warnings():
            simplefilter('ignore', ConvergenceWarning)
            params, opt_data, E_out = self._col_vem_loop(X=X, resampling_step = resampling_step, random_state =random_state)
        #
        tracking_data = self.compute_tracking_data(X, E_out)
        #current_loss = tracking_data['loss_val']
        #current_icl = tracking_data['icl']
        
        # track data
        for k in tracking_data.keys():
            if k not in history.keys():
                history[k] = []
            history[k].append(tracking_data[k])
        
        prev_loss = deepcopy(current_loss)
        
        X = self._impute_missing_data(X=X, random_state= random_state)
        
        params = self._get_parameters()

        opt_data = {#'loss_val': current_loss,
                    'icl': self.computeICL(X),
                    'n_steps': step,
                    'runtime': time.time() - start_time,
                    'success': True,  # subclasses may have failed EM loop
                    'history': history}

        return params, opt_data
  
    def _best_vem_loop(self, X):
        """
        Runs the EM algorithm from multiple initalizations and picks the best solution.
        """

        rng = np.random.default_rng(seed=self.random_state)
        
        init_seeds = np.array([rng.integers(low=0, high= 2**32 -1, size=1).item() for _ in range(self.n_init)])
        
        # lower bounds for each initialization
        init_icls = []
        
        models = []
        for i in range(self.n_init):
            # initialize parameters if not warm starting
            self.initialize_parameters(X, random_state=init_seeds[i])
            
            X = self._impute_missing_data(X, random_state=init_seeds[i])
            
            # EM loop
            with catch_warnings():
                simplefilter('ignore', ConvergenceWarning)
                params, opt_data = self._vem_loop(X=X, random_state = init_seeds[i])
  
            # update parameters if this initialization is better
            #loss_val = opt_data['loss_val']
            icl = opt_data['icl']
            
            #if 'success' in opt_data.keys():
            #    success = opt_data['success']
            #else:
            #    success = True

            if i == 0 or (icl > max(icl) and success):
                best_params = params
                best_opt_data = opt_data
                best_opt_data['init'] = i
                best_opt_data['random_state'] = init_seeds[i]

            init_icls.append(icl)
        
        best_opt_data['init_icls'] = init_icls

        #if not best_opt_data['converged'] and self.verbosity >= 2:
           # warn('Best EM initalization, {} did not'
               #  'converge'.format(best_opt_data['init']), ConvergenceWarning)

        return best_params, best_opt_data

    def _best_sem_loop(self, X):
        """
        Runs the EM algorithm from multiple initalizations and picks the best solution.
        """

        rng = np.random.default_rng(seed=self.random_state)
        
        init_seeds = np.array([rng.integers(low=0, high= 2**32 -1, size=1).item() for _ in range(self.n_init)])
        
        # lower bounds for each initialization
        init_icls = []
        
        models = []
        for i in range(self.n_init):
            # initialize parameters if not warm starting
            self.initialize_parameters(X, random_state=init_seeds[i])
            
            X = self._impute_missing_data(X, random_state=init_seeds[i])
            
            # EM loop
            with catch_warnings():
                simplefilter('ignore', ConvergenceWarning)
                params, opt_data = self._sem_loop(X=X, random_state = init_seeds[i])
  
            # update parameters if this initialization is better
            #loss_val = opt_data['loss_val']
            icl = opt_data['icl']
            
            #if 'success' in opt_data.keys():
            #    success = opt_data['success']
            #else:
            #    success = True

            if i == 0 or (icl > max(icl) and success):
                best_params = params
                best_opt_data = opt_data
                best_opt_data['init'] = i
                best_opt_data['random_state'] = init_seeds[i]

            init_icls.append(icl)
        
        best_opt_data['init_icls'] = init_icls

        #if not best_opt_data['converged'] and self.verbosity >= 2:
           # warn('Best EM initalization, {} did not'
               #  'converge'.format(best_opt_data['init']), ConvergenceWarning)

        return best_params, best_opt_data


def _check_X(X, n_row_components=None, n_col_components=None, ensure_min_samples=1):
    """Check the input data X.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    n_components : int
    Returns
    -------
    X : array, shape (n_samples, n_features)
    """
    #X = check_array(X, dtype=[np.float64, np.float32],
    #                ensure_min_samples=ensure_min_samples)
    if n_row_components is not None and X.shape[0] < n_row_components:
        raise ValueError('Expected n_samples >= n_row_components '
                         'but got n_row_components = %d, n_samples = %d'
                         % (n_row_components, X.shape[0]))
    if n_col_components is not None and X.shape[1] < n_col_components:
        raise ValueError('Expected n_samples >= n_row_components '
                         'but got n_col_components = %d, n_features = %d'
                         % (n_col_components, X.shape[1]))
    return X


_em_docs = dict(
    em_param_docs=dedent("""\
    max_n_steps: int
        Maximum number of EM steps.

    abs_tol: float, None
        Absolute tolerance for EM convergence.

    rel_tol: float, None
        (optional) Relative tolerance for EM convergence.

    n_init: int
        Number of random EM initializations.

    init_params_method: str
        How to initalize the cluster parameters e.g. kmeans.

    init_params_value: None, list
        (optional) User provided value used to initalize the cluster parameters.

    init_weights_method: str
        How to initialize the cluster weights.

    init_weights_value: None, array-like
        (optional) User provided value used to initalize the cluster weights.

    random_state: None, int
        (optional) Random seed for initalization.

    verbosity: int
        How verbose the print out should be (lower means quieter).

    history_tracking: int
        How much optimization data to track as the EM algorithm progresses.
    """)
)
