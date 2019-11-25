import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.activations import softmax
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.utils import Progbar
from scipy.stats import norm
import numpy as np
import integrators

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
    distribution,
    q0,
    n_samples=800,
    path_len=1,
    step_size=0.5, 
    burnin=200,
    stage=1):
  
  momentum_distribution = norm(0,1)

  samples = []
  n_accept = 0

  size = n_samples + burnin

  progress_bar = Progbar(size)

  q_old = q0

  for i in range(size):

    p_old = [momentum_distribution.rvs(size=q_i.shape) for q_i in q_old]

    if stage == 1:
        p_new,q_new = integrators.leapfrog(
            p_old,q_old,distribution,path_len,step_size)
    elif stage == 2:
        p_new,q_new = integrators.two_stage_sympletic(
            p_old,q_old,distribution,path_len,step_size)
    elif stage == 3:
        p_new,q_new = integrators.three_stage_symplectic(
            p_old,q_old,distribution,path_len,step_size)
    
    if should_keep_new_sample(p_old, q_old, p_new,q_new, distribution):
      accept = 1

      q_old = q_new
      if i >= burnin:
        samples.append([q_i.copy() for q_i in q_new])
        
        n_accept += 1
    else:
      accept = 0

      if i >= burnin:
        samples.append([q_i.copy() for q_i in q_old])

    progress_bar.update(i+1, values=[ ("acceptance_rate", accept) ])

  return samples, n_accept / len(samples)
