import numpy as np
from tensorflow.keras.utils import Progbar

def eval_MAP(distribution,samples,X_eval,y_eval):

    samples_layers = [[] for _ in range(len(samples[0]))]
    for _, sample in enumerate(samples):
        for layer_idx, l in enumerate(sample):
            samples_layers[layer_idx].append(l)

    samples_map = [ np.average(l,axis=0) for l in samples_layers ]

    y_probs = distribution.probs(X_eval,samples_map)
    y_preds = np.argmax(y_probs, axis=1)

    n_correct = 0
    for j in range(len(y_preds)):
        n_correct += y_eval[j,y_preds[j]]

    return n_correct / y_eval.shape[0]

def eval_majority(distribution, samples, X_eval, y_eval):
    
    print("Evaluating Majority")
    progbar = Progbar(len(samples))
    y_preds_count = np.zeros_like(y_eval)
    for i,sample in enumerate(samples):
        y_probs = distribution.probs(X_eval, sample)
        y_preds = np.argmax(y_probs, axis=1)

        for j in range(y_preds.shape[0]):
            y_preds_count[j,y_preds[j]] += 1

        progbar.update(i+1)
    
    y_preds_majority = np.argmax(y_preds_count,axis=1)

    n_correct = 0
    for j in range(len(y_preds)):
        n_correct += y_eval[j,y_preds_majority[j]]

    return n_correct / X_eval.shape[0]

def eval_average_accuary(distribution, samples, X_eval, y_eval):

    print("Evaluating Average")
    progbar = Progbar(len(samples))
    y_correct_count = np.zeros( y_eval.shape[0] )
    for i,sample in enumerate(samples):
        y_probs = distribution.probs(X_eval, sample)
        y_preds = np.argmax(y_probs, axis=1)

        for j in range(y_preds.shape[0]):
            y_correct_count[j] += y_eval[j,y_preds[j]]

        progbar.update(i+1)

    return np.average(y_correct_count / len(samples))