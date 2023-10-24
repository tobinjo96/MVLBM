import numpy as np
from tqdm import tqdm
from scipy.stats.distributions import chi2
from sklearn.metrics import adjusted_rand_score


def optimize_over_pi_clust(logphi1, logphi2, row, col, maxiter = 1000, stepsize = 0.01):
  if logphi1.shape[0] != logphi2.shape[0]:
    raise ValueError
  
  n, k1 = logphi1.shape
  k2 = logphi2.shape[1]
  
  Pi = np.ones((k1, k2)) * (1 / (k1 * k2))
  
  obj = np.ones((maxiter)) * np.nan
  
  logphi1_standard = logphi1 - logphi1.max(1)[:, np.newaxis]
  
  normalized_phi1 = np.exp(logphi1_standard) / np.exp(logphi1_standard).sum(1)[:, np.newaxis]
  logphi2_standard = logphi2 - logphi2.max(1)[:, np.newaxis]
  normalized_phi2 = np.exp(logphi2_standard) / np.exp(logphi2_standard).sum(1)[:, np.newaxis]
  
  phi_3D = np.repeat(normalized_phi1[:, :, np.newaxis], k2, axis=2) * np.transpose(np.repeat(normalized_phi2.T[np.newaxis, :, :], k1, axis=0), (2, 0, 1))
  
  for j in range(maxiter):
    phipi_3D = np.repeat(Pi[np.newaxis, :, :], n, axis = 0) * phi_3D
    
    objVect = phipi_3D.sum((1, 2))
    obj[j] = - np.log(objVect).sum()
    
    if np.abs(obj[j] - obj[j-1]) / np.min((obj[j], obj[j - 1])) < 1e-10 and j > 0:
      break
    
    G = normalized_phi1.T.dot(normalized_phi2/objVect[:, np.newaxis])
    
    M = Pi * np.exp(stepsize * G - 1)
    if M.sum() == np.nan:
      raise ValueError
    
    u = np.ones(k1)
    v = np.ones(k2)
    k = M/row[:, np.newaxis]
    
    for i in range(100):
      uprev = u
      u = 1/k.dot(col/M.T.dot(u))
      
      if np.max(np.abs(uprev - u)/np.min((uprev, u))) < 1e-6 and j > 0:
        break
    
    v = col/M.T.dot(u)
    
    Pi = np.diag(u).dot(M).dot(np.diag(v))
  
  obj -= np.log(np.exp(logphi1_standard).sum(1)).sum() + np.log(np.exp(logphi2_standard).sum(1)).sum() +  np.max(logphi1, 1).sum() + np.max(logphi2, 1).sum()
  
  return Pi, obj




def test_p_val(model_1, model_2, data_1, data_2, B = 200, maxiter = 1000, stepsize = 0.01, random_state = None):
  if np.isnan(data_1).sum() > 0:
    data_1 = model_1._impute_missing_data(data_1, random_state)
  
  logphi_1 = model_1.comp_row_log_probs(data_1)
  
  if np.isnan(data_2).sum() > 0:
    data_2 = model_2._impute_missing_data(data_2, random_state)
  
  logphi_2 = model_2.comp_row_log_probs(data_2)
  
  pi_1 = model_1.row_weights_
  pi_2 = model_2.row_weights_
  
  Pi_est, obj = optimize_over_pi_clust(logphi_1, logphi_2, pi_1, pi_2, maxiter = 1000, stepsize = stepsize)
  
  nullLogLik = model_1.row_log_probs(data_1).sum() + model_2.row_log_probs(data_2).sum()
  
  # Compute pseudo likelihood ratio test statistic
  logLambda = -np.min(obj[~np.isnan(obj)]) - nullLogLik
  PLRstat = 2*logLambda
  
  PLRstatperm = np.zeros(B)
  b = 0
  pbar = tqdm(total=B)
  while b < B:
    np.random.seed(random_state)
    np.random.shuffle(logphi_2)
    try:
      Piest, objperm = optimize_over_pi_clust(logphi_1, logphi_2, pi_1, pi_2, maxiter = 500, stepsize = stepsize)
      logLambdaperm = -np.min(objperm[~np.isnan(objperm)]) - nullLogLik
      PLRstatperm[b] = 2*logLambdaperm
      b += 1
      pbar.update(1)
    except:
      continue
  
  return Pi_est, np.mean((PLRstatperm[range(B)] >= PLRstat).astype(int))
   

def test_p_val_G(model_1, model_2, random_state = None):
  y_1 = model_1._get_resps()['row_resps'].argmax(1) 
  y_2 = model_2._get_resps()['row_resps'].argmax(1) 
  
  bin_1 = y_1.max() + 1
  bin_2 = y_1.max() + 1
  
  table = np.bincount(bin_1 * y_1 + y_2, minlength = bin_1 * bin_2).reshape((bin_1, bin_2))
  n = len(y_1)
  G = 0
  for i in range(bin_1):
    for j in range(bin_2):
      G += table[i, j] * (np.log((n * table[i, j]) / table[i, :].sum() * table[:, j].sum()))
  
  G *= 2
  
  return G, chi2.sf(G, (bin_1 - 1) * (bin_2 -1))


def test_p_val_ari(model_1, model_2, B = 200, random_state = None):
  y_1 = model_1._get_resps()['row_resps'].argmax(1) 
  y_2 = model_2._get_resps()['row_resps'].argmax(1) 
  
  ari_stat = adjusted_rand_score(y_1, y_2)
  
  aristatperm = np.zeros(B)
  b = 0
  pbar = tqdm(total=B)
  while b < B:
    np.random.seed(random_state)
    np.random.shuffle(y_2)
    try:
      aristatperm[b] = adjusted_rand_score(y_1, y_2)
      b += 1
      pbar.update(1)
    except:
      continue
  
  return ari_stat, np.mean((aristatperm[range(B)] >= ari_stat).astype(int))

def test_p_val_ALL(model_1, model_2, data_1, data_2, B = 200, maxiter = 1000, stepsize = 0.01, random_state = None):
  if np.isnan(data_1).sum() > 0:
    data_1 = model_1._impute_missing_data(data_1, random_state)
  
  logphi_1 = model_1.comp_row_log_probs(data_1)
  
  if np.isnan(data_2).sum() > 0:
    data_2 = model_2._impute_missing_data(data_2, random_state)
  
  logphi_2 = model_2.comp_row_log_probs(data_2)
  
  y_1 = model_1._get_resps()['row_resps'].argmax(1) 
  y_2 = model_2._get_resps()['row_resps'].argmax(1) 
  
  pi_1 = model_1.row_weights_
  pi_2 = model_2.row_weights_
  
  Pi_est, obj = optimize_over_pi_clust(logphi_1, logphi_2, pi_1, pi_2, maxiter = 1000, stepsize = stepsize)
  
  nullLogLik = model_1.row_log_probs(data_1).sum() + model_2.row_log_probs(data_2).sum()
  
  # Compute pseudo likelihood ratio test statistic
  logLambda = -np.min(obj[~np.isnan(obj)]) - nullLogLik
  PLRstat = 2*logLambda
  
  ari_stat = adjusted_rand_score(y_1, y_2)
  
  aristatperm = np.zeros(B)
  
  PLRstatperm = np.zeros(B)
  b = 0
  pbar = tqdm(total=B)
  while b < B:
    np.random.seed(random_state)
    np.random.shuffle(logphi_2)
    np.random.seed(random_state)
    np.random.shuffle(y_2)
    try:
      Piest, objperm = optimize_over_pi_clust(logphi_1, logphi_2, pi_1, pi_2, maxiter = 500, stepsize = stepsize)
      logLambdaperm = -np.min(objperm[~np.isnan(objperm)]) - nullLogLik
      PLRstatperm[b] = 2*logLambdaperm
      aristatperm[b] = adjusted_rand_score(y_1, y_2)
      b += 1
      pbar.update(1)
    except:
      continue
  
  bin_1 = y_1.max() + 1
  bin_2 = y_2.max() + 1
  
  table = np.bincount(bin_1 * y_1 + y_2, minlength = bin_1 * bin_2).reshape((bin_1, bin_2))
  n = len(y_1)
  G = 0
  for i in range(bin_1):
    for j in range(bin_2):
      G += table[i, j] * (np.log((n * table[i, j]) / (table[i, :].sum() * table[:, j].sum())))
  
  G *= 2
  
  return np.mean((PLRstatperm[range(B)] >= PLRstat).astype(int)), chi2.sf(G, (bin_1 - 1) * (bin_2 -1)), np.mean((aristatperm[range(B)] >= ari_stat).astype(int))











