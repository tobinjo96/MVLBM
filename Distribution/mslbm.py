from sklearn.base import BaseEstimator, DensityMixin
from textwrap import dedent
from msbase import MultiSetBlockModel
from mvbase import MultiViewMixtureModelMixin, MultiViewEMixin
from base import _em_docs
from copy import deepcopy


class MSLBM(MultiSetBlockModel,
           BaseEstimator, DensityMixin):

    def __init__(self,
                 col_indices = None,
                 base_set_models=None,
                 max_n_steps=200,
                 row_max_n_steps=1,
                 col_max_n_steps=1,
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

        MultiSetBlockModel.__init__(self,
                                 col_indices = col_indices,
                                 max_n_steps=max_n_steps,
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

        self.base_set_models = base_set_models
        self.col_indices = col_indices
        self.n_sets = len(col_indices)
        self.set_models_ = [deepcopy(self.base_set_models[v])
                             for v in range(self.n_sets)]
            
        row_components = [vm.n_row_components for vm in self.base_set_models]
        assert row_components.count(row_components[0]) == len(row_components)
        self.n_row_components = row_components[0]

MSLBM.__doc__ = dedent("""\
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
