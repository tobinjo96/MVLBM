import numpy as np
from itertools import product
from numbers import Number
from scipy.special import logsumexp

from mvlbm import MVLBM


class MaskedMVLBM(MVLBM):
    """
    A multi-view mixture model where some entries of Pi are set to zero.
    """

    def _pre_fit_setup(self):
        """
        Subclasses may call this before running _fit()
        """
        self.n_zeroed_row_comps_ = 0

    @property
    def n_row_components(self):
        """
        Returns the total number of clusters.
        """
        if hasattr(self, 'zero_mask_') and self.zero_mask_ is not None:
            return (~self.zero_mask_).ravel().sum()

        else:
            return np.product(self.n_view_row_components)

    @property
    def row_weights_mat_(self):
        """
        Returns weights as a matrix.
        """

        if hasattr(self, 'row_weights_') and self.row_weights_ is not None:

            row_weights_mat_ = np.zeros(self.n_view_row_components)
            for k in range(self.n_row_components):
                # idx_0, idx_1 = self._get_view_clust_idx(k)
                # weights_mat[idx_0, idx_1] = self.weights_[k]
                view_idxs = self._get_view_clust_idx(k)
                row_weights_mat_[view_idxs] = self.row_weights_[k]

            return row_weights_mat_
   
    @property
    def row_resps_mat_(self):
        """
        Returns weights as a matrix.
        """
        # TODO: does this mess up check_is_fitted()
        n_samples = self.row_resps_.shape[0]
        if hasattr(self, 'row_resps_') and self.row_resps_ is not None:
            row_resps_reshape = [n_samples]
            row_resps_reshape.extend(self.n_view_row_components)
            row_resps_mat_ = np.zeros(row_resps_reshape)
            for k in range(self.n_row_components):
                # idx_0, idx_1 = self._get_view_clust_idx(k)
                # weights_mat[idx_0, idx_1] = self.weights_[k]
                view_idxs = self._get_view_clust_idx(k)
                view_idxs = tuple((Ellipsis, *view_idxs))
                row_resps_mat_[view_idxs] = self.row_resps_[:, k]

            return row_resps_mat_
    
    def _get_view_clust_idx(self, k):
        """Retuns the view cluster indices for each view for an overall cluster index.

        Returns
        -------
        view_idxs : array, shape (n_views, )
        """

        if isinstance(k, Number):
            return self._get_view_clust_idx_for_masked(int(k))

        else:
            view_idxs = [[] for v in range(self.n_views)]
            for idx in k:
                _view_idxs = self._get_view_clust_idx_for_masked(int(idx))
                for v in range(self.n_views):
                    view_idxs[v].append(_view_idxs[v])
            return tuple(np.array(view_idxs[v]) for v in range(self.n_views))

            # row_idxs, col_idxs = [], []
            # for idx in k:
            #     r, c = self._get_view_clust_idx_for_masked(int(idx))
            #     row_idxs.append(r)
            #     col_idxs.append(c)

            # return np.array(row_idxs), np.array(col_idxs)

    def _get_view_clust_idx_for_masked(self, k):

        assert k < self.n_row_components

        idx = -1
        ranges = tuple(range(self.n_view_row_components[v])
                       for v in range(self.n_views))

        for view_idxs in product(*ranges):
            # don't count components which are zeroed out
            if not self.zero_mask_[view_idxs]:
                idx += 1

            if k == idx:
                return view_idxs

        # idx = -1
        # for idx_0, idx_1 in product(range(self.n_view_components[0]),
        #                             range(self.n_view_components[1])):
        #     # don't count components which are zeroed out
        #     if not self.zero_mask_[idx_0, idx_1]:
        #         idx += 1

        #     if k == idx:
        #         return idx_0, idx_1

        raise ValueError('No components found.')

    def _get_overall_clust_idx(self, view_idxs):

        assert not self.zero_mask_[view_idxs]

        for k in range(self.n_row_components):
            _view_idxs = self._get_view_clust_idx(k)

            if all(_view_idxs[v] == view_idxs[v]
                   for v in range(self.n_views)):
                return k

    def _zero_initialization(self, init_params):
        zero_mask = np.zeros(shape=self.n_view_row_components).astype(bool)
        init_params['zero_mask'] = zero_mask
        removed_mask = np.zeros(shape=self.n_view_row_components).astype(bool)
        init_params['removed_mask'] = removed_mask
        self.n_zeroed_row_comps_ = 0
        for v in range(self.n_views):
            if hasattr(self, 'view_models_'):
              init_params['views'][v] = self.view_models_[v]._zero_initialization(init_params['views'][v])
            else:
              init_params['views'][v] = self.base_view_models[v]._zero_initialization(init_params['views'][v])

        return init_params

    def _get_parameters(self):
        view_params = []
        for v in range(self.n_views):
            view_params.append(self.view_models_[v]._get_parameters())

        return {'views': view_params,
                'row_weights': self.row_weights_,
                'zero_mask': self.zero_mask_,
                'removed_mask': self.removed_mask_}

    def _set_parameters(self, params):
        if 'views' in params.keys():
            for v in range(self.n_views):
                self.view_models_[v]._set_parameters(params['views'][v])

        # TODO: move this after weights
        if 'zero_mask' in params.keys():
            self.zero_mask_ = params['zero_mask']
        
        if 'removed_mask' in params.keys():
            self.removed_mask_ = params['removed_mask']
        
        if 'row_weights' in params.keys():
            self.row_weights_ = params['row_weights']

            # make sure each view's weights_ is the marginal of the weights
            for v in range(self.n_views):
                ax_to_sum = tuple([a for a in range(self.n_views) if a != v])
                view_row_weights = np.sum(self.row_weights_mat_, axis=ax_to_sum)
                self.view_models_[v]._set_parameters({'row_weights': view_row_weights})

        # drop components with 0 weight
        idxs2drop = np.where(self.row_weights_ == 0.0)[0]
        if len(idxs2drop) > 0:
            self.drop_component(idxs2drop)

        # TODO: something crazy is happening -- the code won't reach here
        # if 'zero_mask' in params.keys():
        #     print('here')
        #     self.zero_mask_ = params['zero_mask']
    
    def _row_m_step_clust_params(self, X, log_resps):
        """
        M step. Each view's cluster parameters can be updated independently.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        log_resps : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or respsonsibilities) of
            the point of each sample in X.
        """
        # TODO: document this as it is a critical step

        # for each view-cluster pair, which columns of log_resps to logsumsxp
        vc_axes2sum = [[[] for c in range(self.view_models_[v].n_row_components)]
                       for v in range(self.n_views)]

        for k in range(self.n_row_components):
            view_idxs = self._get_view_clust_idx(k)
            for v in range(self.n_views):
                vc_axes2sum[v][view_idxs[v]].append(k)

            # idx_0, idx_1 = self._get_view_clust_idx(k)
            # vc_axes2sum[0][idx_0].append(k)
            # vc_axes2sum[1][idx_1].append(k)

        view_params = [None for v in range(self.n_views)]

        for v in range(self.n_views):
            view_log_resps = []
            # for each view-component logsumexp the respsonsibilities
            for c in range(self.view_models_[v].n_row_components):
                axes2sum = vc_axes2sum[v][c]
                view_log_resps.append(logsumexp(log_resps[:, axes2sum], axis=1))
            view_log_resps = np.array(view_log_resps).T

            view_params[v] = self.view_models_[v].\
                _m_step_clust_params(X[v], view_log_resps, self.view_models_[v].col_resps_)

        return view_params

    def _row_sm_step_clust_params(self, X, resps):
        """
        M step. Each view's cluster parameters can be updated independently.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        log_resps : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or respsonsibilities) of
            the point of each sample in X.
        """
        # TODO: document this as it is a critical step

        # for each view-cluster pair, which columns of log_resps to logsumsxp
        vc_axes2sum = [[[] for c in range(self.view_models_[v].n_row_components)]
                       for v in range(self.n_views)]

        for k in range(self.n_row_components):
            view_idxs = self._get_view_clust_idx(k)
            for v in range(self.n_views):
                vc_axes2sum[v][view_idxs[v]].append(k)

            # idx_0, idx_1 = self._get_view_clust_idx(k)
            # vc_axes2sum[0][idx_0].append(k)
            # vc_axes2sum[1][idx_1].append(k)

        view_params = [None for v in range(self.n_views)]

        for v in range(self.n_views):
            view_resps = []
            # for each view-component logsumexp the respsonsibilities
            for c in range(self.view_models_[v].n_row_components):
                axes2sum = vc_axes2sum[v][c]
                view_resps.append(np.sum(resps[:, axes2sum], axis=1))
            view_resps = np.array(view_resps).T

            view_params[v] = self.view_models_[v].\
                _sm_step_clust_params(X[v], view_resps, self.view_models_[v].col_resps_)

        return view_params
    
    def _combine_SEM_params(self, params_list):
        out_params = {}
        input_shape = tuple([self.base_view_models[v].n_row_components for v in range(self.n_views)])
        _final_mask = params_list[-1]['removed_mask']
        row_weights_list = []
        for params in params_list:
            step_row_weights = params['row_weights']
            step_retained_mask = np.invert(params['removed_mask'])
            inter_weights = np.zeros(input_shape, dtype=float)
            inter_weights[step_retained_mask] = step_row_weights.reshape(-1)
            inter_weights[_final_mask] = np.zeros(np.sum(_final_mask)).reshape(-1)
            #np.putmask(inter_weights, step_retained_mask, step_row_weights)
            #np.putmask(inter_weights, _final_mask, np.zeros(np.sum(_final_mask)))
            inter_weights /= np.sum(inter_weights)
            row_weights_list.append(inter_weights[np.invert(_final_mask)])
            
        row_weights = np.average(np.stack(row_weights_list), axis = 0)
        row_weights /= row_weights.sum()
        out_params['row_weights'] = row_weights
        
        out_params['views'] = []
        for v in range(self.n_views):
          view_params_list = [params['views'][v] for params in params_list]
          params = self.view_models_[v]._combine_SEM_params(view_params_list)
          out_params['views'].append(params)
        
        return out_params
    
    def drop_component(self, comps):
        """
        Drops a component or components from the model.

        Parameters
        ----------
        comps: int, list of ints
            Which component(s) to drop. On the scal of overall indices.
        """

        # TODO: re-write using _set_parameters
        if isinstance(comps, Number):
            comps = [comps]

        self.n_zeroed_row_comps_ += len(comps)

        # sort componets in decreasing order so that lower indicies
        # are preserved after dropping higher indices
        comps = np.sort(comps)[::-1]

        overall_comps2drop = []
        view_comps2drop = [[] for v in range(self.n_views)]
        # view_0_comps2drop = []
        # view_1_comps2drop = []
        #remaining_idx = np.where(self.removed_mask_ == 0)
        removed_axes = [np.where(np.mean(self.removed_mask_, axis = tuple([i for i in range(self.n_views) if i != v])) == 1)[0] for v in range(self.n_views)]
        for k in comps:
            # idx_0, idx_1 = self._get_view_clust_idx(k)
            view_idxs = self._get_view_clust_idx(k)
            # don't drop components which are already zero
            if not self.zero_mask_[view_idxs]:
                self.zero_mask_[view_idxs] = True
                remove_idx = tuple([view_idxs[i] + np.sum(view_idxs[i] + len(removed_axes[i]) > removed_axes[i]) for i in range(self.n_views)])
                self.removed_mask_[remove_idx] = True
                overall_comps2drop.append(k)
                for v in range(self.n_views):
                    # TODO: check this
                    meow = self.zero_mask_.take(indices=view_idxs[v], axis=v)
                    if np.mean(meow) == 1:
                        view_comps2drop[v].append(view_idxs[v])

                # # if row idx_0 is entirely zero, drop component
                # # idx_0 from the view 0 model
                # if np.mean(self.zero_mask_[idx_0, :]) == 1:
                #     view_0_comps2drop.append(idx_0)

                # # similarly drop zero columns
                # if np.mean(self.zero_mask_[:, idx_1]) == 1:
                #     view_1_comps2drop.append(idx_1)

        # drop entries from weights_ and re-normalize
        print(overall_comps2drop)
        self.row_weights_ = np.delete(self.row_weights_, overall_comps2drop)
        self.row_weights_ = self.row_weights_ / sum(self.row_weights_)
        
        #drop entries from row_resps_
        self.row_resps_ = np.delete(self.row_resps_, [a for a in overall_comps2drop], 1)
        self.row_resps_ = self.row_resps_ / self.row_resps_.sum(1)[:, np.newaxis]
        
        print(view_comps2drop)
        for v in range(self.n_views):
            self.view_models_[v].drop_component(view_comps2drop[v])
            print('MASKED:', self.view_models_[v].n_row_components)
            self.zero_mask_ = np.delete(self.zero_mask_,
                                        view_comps2drop[v],
                                        axis=v)

    def reorder_components(self, new_idxs):
        """
        Re-orders the components

        Parameters
        ----------
        new_idxs_0: list
            List of the new index ordering for view 0
            i.e. new_idxs_0[0] maps old index 0 to its new index.

        new_idxs_0: list
            List of the new index ordering for view 1
            i.e. new_idxs_0[0] maps old index 0 to its new index.

        """

        for v in range(self.n_views):
            # TODO: re-write using _set_parameters()
            assert set(new_idxs[v]) == set(range(self.n_view_row_components[v]))

        # for new overall index ordering
        new_entry_ordering = []
        old2new = [{old: new for new, old in enumerate(new_idxs[v])}
                   for v in range(self.n_views)]

        for k in range(self.n_row_components):
            old_view_idxs = self._get_view_clust_idx(k)

            new_view_idxs = [old2new[v][old_view_idxs[v]]
                             for v in range(self.n_views)]

            new_entry_ordering.append(new_view_idxs)

        # reorder view cluster parameters
        for v in range(self.n_views):
            self.view_models_[v].reorder_components(new_idxs[v])

        # re-order zero mask
        for v in range(self.n_views):
            self.zero_mask_ = self.zero_mask_.take(indices=new_idxs[v],
                                                   axis=v)

        # get new overall ordering and re-order weights_
        new_idxs_overall = []
        for new_idxs in new_entry_ordering:
            k_new = self._get_overall_clust_idx(new_idxs)
            new_idxs_overall.append(k_new)

        self.row_weights_ = self.row_weights_[new_idxs_overall]

        return self

    def _n_row_weight_parameters(self):
        tot = np.product(self.n_view_row_components)
        n_zeros = self.zero_mask_.sum()

        return tot - n_zeros - 1

    # def _obs_nll(self, X):
    #     
    #     mask = ~(self.zero_mask_.ravel())
    #     obs_nll = -np.sum(self.row_resps_[:, mask] * np.log(self.row_resps_[:, mask])) + self.row_resps_.sum(0) @ np.log(self.row_weights_)
    #     
    #     for v in range(self.n_views):
    #         nk, nl = self.view_models_[v].n_row_components, self.view_models_[v].n_col_components
    #         indices_ones = np.where(X[v] == 1)
    #         ax_to_sum = tuple([a + 1 for a in range(self.n_views) if a != v])
    #         view_row_resps = np.sum(self.row_resps_mat_, axis=ax_to_sum)
    #         
    #         obs_nll += ((np.sum(self.view_models_[v].col_resps_ * np.log(self.view_models_[v].col_resps_)) +
    #           self.view_models_[v].col_resps_.sum(0) @ np.log(self.view_models_[v].col_weights_).T )
    #         
    #         + (
    #             view_row_resps[indices_ones[0]].reshape(-1, nk, 1)
    #             * self.view_models_[v].col_resps_[indices_ones[1]].reshape(-1, 1, nl)
    #             * (
    #                 np.log(self.view_models_[v].probability_.reshape(1, nk, nl))
    #                 - np.log(1 - self.view_models_[v].probability_).reshape(1, nk, nl)
    #             )
    #         ).sum()
    #         + (view_row_resps.sum(0) @ np.log(1 - self.view_models_[v].probability_) @ self.view_models_[v].col_resps_.sum(0))
    #     )
    #     
    #     return obs_nll
