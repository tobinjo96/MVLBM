import numpy as np
from warnings import warn
from copy import deepcopy
from textwrap import dedent

from base import _em_docs
from maskedmvlbm import MaskedMVLBM


class LogPenMVLBM(MaskedMVLBM):

    def __init__(self,
                 pen=None,
                 delta=1e-6,
                 base_view_models=None,
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
        self.pen = pen
        self.delta = delta

    def _row_m_step_weights(self, X, log_resps):
        """
        M step for the cluster membership weights.

        log_resps : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or respsonsibilities) of
            the point of each sample in X.
        """

        n_samples = log_resps.shape[0]
        resps = np.exp(log_resps)
        nk = resps.sum(axis=0) + 10 * np.finfo(resps.dtype).eps
        a = nk / n_samples  # normalize so sum(a) == 1

        if self.pen is None:
            return a

        # compute soft thresholding operation
        a_thresh, idxs_dropped = soft_thresh(a, pen=self.pen)

        # if all entries are dropped then set largest entry to 0
        if len(idxs_dropped) == len(a):
            idx_max = np.argmax(a)
            a_thresh[idx_max] = 1.0
            idxs_dropped = np.delete(idxs_dropped, idx_max)

            warn('Soft-thresholding attempting to drop every component.'
                 'Retaining component with largest weight.')

        # TODO: this violates setting the parameters using
        # _set_parameters() -- need to figure out what to do with
        # drop_component e.g. make drop_component return the new zero mask

        # set weights to soft-thresholded values
        # self.weights_ = a_thresh

        # drop any components which got zeroed out
        # if len(idxs_dropped) > 0:
        #     self.drop_component(idxs_dropped)

        return a_thresh

    def _row_sm_step_weights(self, X, resps):
        """
        M step for the cluster membership weights.

        log_resps : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or respsonsibilities) of
            the point of each sample in X.
        """

        n_samples = resps.shape[0]
        nk = resps.sum(axis=0) + 10 * np.finfo(resps.dtype).eps
        a = nk / n_samples  # normalize so sum(a) == 1

        if self.pen is None:
            return a

        # compute soft thresholding operation
        a_thresh, idxs_dropped = soft_thresh(a, pen=self.pen)

        # if all entries are dropped then set largest entry to 0
        if len(idxs_dropped) == len(a):
            idx_max = np.argmax(a)
            a_thresh[idx_max] = 1.0
            idxs_dropped = np.delete(idxs_dropped, idx_max)

            warn('Soft-thresholding attempting to drop every component.'
                 'Retaining component with largest weight.')

        # TODO: this violates setting the parameters using
        # _set_parameters() -- need to figure out what to do with
        # drop_component e.g. make drop_component return the new zero mask

        # set weights to soft-thresholded values
        # self.weights_ = a_thresh

        # drop any components which got zeroed out
        # if len(idxs_dropped) > 0:
        #     self.drop_component(idxs_dropped)

        return a_thresh

    def compute_tracking_data(self, X, E_out):

        out = {}

        if 'obs_nll' in E_out.keys():
            #out['obs_nll'] = E_out['obs_nll']
            out['icl'] = E_out['icl']
        else:
            #out['obs_nll'] = - self.score(X)
            out['icl'] = self.computeICL(X)
        

        out['n_zeroed_row_comps'] = deepcopy(self.n_zeroed_row_comps_)

        if self.pen is not None:
            out['log_pen'] = self.pen * \
                (np.log(self.delta + self.row_weights_).sum() +
                 np.log(self.delta) * self.n_zeroed_row_comps_)
        else:
            out['log_pen'] = 0

        #out['loss_val'] = out['obs_nll'] + out['log_pen']

        # maybe track model history
        if self.history_tracking >= 2:
            out['model'] = deepcopy(self._get_parameters())

        return out


LogPenMVLBM.__doc__ = dedent("""\
Log penalized multi-view mixture model fit using an EM algorithm.

Parameters
----------
pen: float
    Penalty value.

delta: float
    Delta value for penalty -- only used for stopping criterion.

base_view_models: list of mixture models
    Mixture models for each view. These should specify the number of view components.

{em_param_docs}

Attributes
----------
weights_

weights_mat_

zero_mask_

metadata_

""".format(**_em_docs))


def soft_thresh(a, pen):
    """
    Computes soft-thresholding operation approximation of log(pi_k + epsilon) penalty.

    a_thresh = max(0, a - pen)

    if epsilon is not None:
        a_thresh[zeroed_entries] = epsilon

    a_thresh = noralize(a_thresh)


    Parameters
    ----------
    a:  array-like, (n_components, )
        The coefficient values.

    pen: float
        The soft-thresholding parameter.


    Output
    ------
    a_thresh, idxs_dropped

    a_thresh: array-like, (n_components, )
        The thresholded coefficient values

    idxs_dropped: array-like
        List of indices set to zero.
    """
    a = np.array(a)

    # which components will be dropped
    drop_mask = a <= pen

    if np.mean(drop_mask) == 1:
        raise warn('Soft-thresholding dropping all components')

    # soft threshold entries of a
    a_soft_thresh = np.clip(a - pen, a_min=0, a_max=None)

    # normalize
    tot = a_soft_thresh.sum()
    a_soft_thresh = a_soft_thresh / tot

    drop_idxs = np.where(drop_mask)[0]
    return a_soft_thresh, drop_idxs
