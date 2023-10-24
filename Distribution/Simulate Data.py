from data_generator import sample_lbm
import numpy as np
import pickle as pkl

#Pis
K = 3
Pi_list = []
for delta in np.arange(0, 1.125, 0.125):
  Pi_list.append(((1-delta)/K**2) * np.ones((K, K)) + (delta/K) *  np.diag(np.ones(K)))


#Params
l1 = 1/(4.6e-7 * 500**2)
l2 = 1/(20.5e-7 * 500**2)
l3 = 1/(4.9e-7 * 500**2)
l4 = 1/(30.0e-7 * 500**2)
l5 = 1/(20.5e-7 * 500**2)
l6 = 1/(1.6e-7 * 500**2)
l7 = 1/(5.5e-7 * 500**2)
l8 = 1/(5.6e-7 * 500**2)
l9 = 1/(14.5e-7 * 500**2)




means1 = np.array([[100, 0.5, -90], [10, -15, -95], [-20, -30, 500]])
covariances1 = np.array([[1, 25, 25], [16, 1, 1], [1, 9, 16]])
col_weights1 = np.array([1/3, 1/3, 1/3])
modes = np.array([[2, 0, 2], [1, 2, 1], [1, 0, 1]])
proportions = np.array([[0.4, 0.2, 0.7], [0.1, 0.5, 0.8], [0.5, 0.8, 0.2]])
col_weights2 = np.array([1/3, 1/3, 1/3])
probability = np.array([[[0.05, 0.05, 0.8, 0.05, 0.05], [0.1, 0.25, 0.3, 0.3, 0.05], [0.1, 0.2, 0.4, 0.2, 0.1]], 
                        [[0.05, 0.1, 0.7, 0.1, 0.05], [0.8, 0.05, 0.05, 0.05, 0.05], [0.4, 0.05, 0.1, 0.05, 0.4]], 
                        [[0.2, 0.5, 0.2, 0.05, 0.05], [0.8, 0.05, 0.05, 0.05, 0.05], [0.05, 0.8, 0.05, 0.05, 0.05]]])
col_weights3 = np.array([1/3, 1/3, 1/3])
rates = np.array([[l1, l2, l3], [l4, l5, l6], [l7, l8, l9]])
col_weights4 = np.array([1/3, 1/3, 1/3])
col_weights = np.array([1/4, 1/4, 1/4, 1/4])


view_params1 = [{'set parameters': [{'means':means1, 'covariances': covariances1, 'col_weights':col_weights1},
                                  {'modes':modes, 'proportions': proportions, 'col_weights':col_weights2}, 
                                  {'probability':probability, 'col_weights':col_weights3},
                                  {'rates': rates, 'col_weights':col_weights4}], 
                                  'col_weights': col_weights}, 
                {'set parameters': [{'means':means1, 'covariances': covariances1, 'col_weights':col_weights1},
                                  {'modes':modes, 'proportions': proportions, 'col_weights':col_weights2}, 
                                  {'probability':probability, 'col_weights':col_weights3},
                                  {'rates': rates, 'col_weights':col_weights4}], 
                                  'col_weights': col_weights}]

means2 = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])
covariances2 = np.ones((3, 3))

view_params2 = [{'set parameters': [{'means':means2, 'covariances': covariances2, 'col_weights':col_weights1},
                                  {'modes':modes, 'proportions': proportions, 'col_weights':col_weights2}, 
                                  {'probability':probability, 'col_weights':col_weights3},
                                  {'rates': rates, 'col_weights':col_weights4}], 
                                  'col_weights': col_weights}, 
                {'set parameters': [{'means':means2, 'covariances': covariances2, 'col_weights':col_weights1},
                                  {'modes':modes, 'proportions': proportions, 'col_weights':col_weights2}, 
                                  {'probability':probability, 'col_weights':col_weights3},
                                  {'rates': rates, 'col_weights':col_weights4}], 
                                  'col_weights': col_weights}]

view_params_list = [view_params1, view_params2]

#Nrow
n_row_samples_list = [300, 1200]

#Ncol
n_col_samples_list = [[240, 240], [1200, 1200]]

#NaNs
na_pct_list = [0, 0.15, 0.35]

#Rngs
random_state_list = range(10)


#Running
i = 0
for Pi in Pi_list:
  j = 0
  for view_params in view_params_list:
    for n_row_samples in n_row_samples_list:
      l = 0
      for n_col_samples in n_col_samples_list:
        for na_pct in na_pct_list:
          for random_state in random_state_list:
            view_data = sample_lbm(view_params, Pi, n_row_samples, n_col_samples, na_pct = na_pct, random_state=random_state)
            with open('Data/Simulated/sim_data_Pi' + str(i) + 'vp' + str(j) + '_nr' + str(n_row_samples) + '_nc' + str(l) + '_na' + str(na_pct) + '_' + str(random_state) + '.pkl', 'wb') as f:
              pkl.dump(view_data, f)
              f.close()
        l += 1
    
    j += 1
  
  i += 1
