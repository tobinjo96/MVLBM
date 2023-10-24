from sklearn.base import BaseEstimator, DensityMixin
from textwrap import dedent

from mvbase import MultiViewMixtureModelMixin, MultiViewEMixin
from base import _em_docs


class MVLBM(MultiViewEMixin, MultiViewMixtureModelMixin,
           BaseEstimator, DensityMixin):

    def __init__(self,
                 base_view_models,
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

        MultiViewEMixin.__init__(self,
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

        self.base_view_models = base_view_models


MVLBM.__doc__ = dedent("""\
Multi-view mixture model fit using an EM algorithm.

Parameters
----------
base_view_models: list of mixture models
    Mixture models for each view. These should specify the number of view components.

{em_param_docs}

Attributes
----------

weights_

weights_mat_

metadata_

""".format(**_em_docs))
