import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.activations import softmax
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.utils import Progbar
from scipy.stats import norm
import numpy as np

def leapfrog_slow(p,q,distribution,path_len,step_size):
    p = [p_i.copy() for p_i in p]
    q = [q_i.copy() for q_i in q]

    assert(len(p) == len(q))

    for _ in range( int(path_len / step_size) ):

        dq = distribution.negative_log_posterior_gradient(q)
        for p_i,dq_i in zip(p,dq):
            p_i += -step_size * dq_i / 2
        
        for p_i,q_i in zip(p,q):
            q_i += step_size * p_i

        dq = distribution.negative_log_posterior_gradient(q)
        for p_i,dq_i in zip(p,dq):
            p_i += -step_size * dq_i / 2
        
    return [-p_i for p_i in p], q

def should_keep_new_sample(p_old, q_old, p_new, q_new, distribution):
    p_neg_log_prob_old = np.sum([ np.sum(p_i**2) / 2 for p_i in p_old ])
    q_neg_log_prob_old = distribution.negative_log_posterior(q_old)
    H_old = p_neg_log_prob_old + q_neg_log_prob_old

    p_neg_log_prob_new = np.sum([ np.sum(p_i**2) / 2 for p_i in p_new ])
    q_neg_log_prob_new = distribution.negative_log_posterior(q_new)
    H_new = p_neg_log_prob_new + q_neg_log_prob_new

    p_accept = min(1.0, np.exp(H_old - H_new))
    if np.random.rand() < p_accept:
        return True
    else:
        return False

def hamiltonian_monte_carlo(
    distribution,q0,
    n_samples=1000,path_len=1,step_size=0.5, burnin=0):
  
  momentum_distribution = norm(0,1)

  samples = []
  n_accept = 0

  size = n_samples + burnin

  progress_bar = Progbar(size+1)

  q_old = q0

  for i in range(size):
    progress_bar.update(i+1)

    p_old = [momentum_distribution.rvs(size=q_i.shape) for q_i in q_old]

    p_new,q_new = leapfrog_slow(
        p_old,q_old,distribution,path_len,step_size)
    
    if should_keep_new_sample(p_old, q_old, p_new,q_new, distribution):
      
      q_old = q_new
      if i >= burnin:
        samples.append([q_i.copy() for q_i in q_new])
        
        n_accept += 1
    else:

      if i >= burnin:
        samples.append([q_i.copy() for q_i in q_old])

  return samples, n_accept / len(samples)
