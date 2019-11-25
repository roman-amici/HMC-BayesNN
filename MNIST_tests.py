from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import HMC
import distributions
import tensorflow as tf
import evaluations
from os import path


def get_mnist(use_bias=True):
    load = fetch_mldata('MNIST Original')
    y,X = load['target'], load['data']
    
    if(use_bias):
        X = np.hstack([X,np.ones( (X.shape[0],1) )])

    enc = OneHotEncoder()

    np.random.seed(18523)
    perm = np.random.permutation(X.shape[0])

    X = X[perm, :] / 255
    y = y[perm]

    X_test = X[:10000,:]
    y_test = y[:10000]
    y_test = enc.fit_transform( np.reshape(y_test, (-1,1) )).toarray()

    X_val = X[10000:20000, :]
    y_val = y[10000:20000]
    y_val = enc.transform(np.reshape(y_val,(-1,1) )).toarray()

    X = X[20000:,:]
    y = y[20000:]
    y = enc.transform( np.reshape(y, (-1,1) )).toarray()

    return X,y,X_val,y_val,X_test,y_test

def log_trial(
    filename,
    input_size,
    path_length,
    step_size,
    work_units,
    n_samples,
    burnin,
    width,
    stage,
    acceptance_rate,
    MAP_accuracy,
    majority_accuracy,
    average_accuracy):

    if not path.exists(filename):
        with open(filename, "w") as f:
            f.write("input size, path length, step size, n samples, burnin, work units, width, stage, acceptance rate, MAP accuracy, majority accuracy, average accuracy\n")
        
    with open(filename, "a") as f:
        f.write(f"{input_size}, {path_length}, {step_size}, {n_samples}, {burnin}, {work_units}, {width}, {stage}, {acceptance_rate}, {MAP_accuracy}, {majority_accuracy}, {average_accuracy}\n")

def acceptance_rate_search(
    filename="tests.txt",
    input_size=2500,
    path_length=1,
    n_samples=800,
    burnin=200):

    X,y,X_val,y_val,_,_ = get_mnist()

    X_sub, y_sub = X[:input_size,:], y[:input_size,:]

    for width in [20,50,100]:
        print("Width:", width)
        ws = [np.random.normal(size=(X.shape[1], width)), np.random.normal( size=(width, y_sub.shape[1])) ]
        tf.reset_default_graph()
        bnn = distributions.Bayesian_NN_Session(X_sub,y_sub, [w.shape for w in ws ])

        for stage in [1,2,3]:
            print("Stage", stage)
            for eps in [.025, .01]:
                print("Step_size:",eps)
                samples,acceptance_rate = HMC.hamiltonian_monte_carlo(
                    bnn, 
                    ws,
                    n_samples,
                    path_length, 
                    eps, 
                    burnin,
                    stage)

                MAP_accuracy = evaluations.eval_MAP(bnn,samples,X_val,y_val)
                majority_accuracy = evaluations.eval_majority(bnn,samples,X_val,y_val)
                average_accuracy = evaluations.eval_average_accuary(bnn,samples,X_val,y_val)
                work_units = stage * (path_length // eps)

                log_trial(
                    filename,
                    input_size,
                    path_length,
                    eps,
                    work_units,
                    n_samples,
                    burnin,
                    width,
                    stage,
                    acceptance_rate,
                    MAP_accuracy,
                    majority_accuracy,
                    average_accuracy
                )

def normalized_search(
    filename="tests.txt",
    input_size=2500,
    path_length=1,
    n_samples=800,
    burnin=200):

    X,y,X_val,y_val,_,_ = get_mnist()

    X_sub, y_sub = X[:input_size,:], y[:input_size,:]

    for width in [20,50,100]:
        print("Width:", width)
        ws = [np.random.normal(size=(X.shape[1], width)), np.random.normal( size=(width, y_sub.shape[1])) ]
        tf.reset_default_graph()
        bnn = distributions.Bayesian_NN_Session(X_sub,y_sub, [w.shape for w in ws ])

        for stage in [1,2,3]:
            print("Stage", stage)
            for eps_base in [.025, .01, .005]:
                eps = eps_base * stage

                print("Step_size:",eps)
                samples,acceptance_rate = HMC.hamiltonian_monte_carlo(
                    bnn, 
                    ws,
                    n_samples,
                    path_length, 
                    eps, 
                    burnin,
                    stage)

                MAP_accuracy = evaluations.eval_MAP(bnn,samples,X_val,y_val)
                majority_accuracy = evaluations.eval_majority(bnn,samples,X_val,y_val)
                average_accuracy = evaluations.eval_average_accuary(bnn,samples,X_val,y_val)
                work_units = stage * (path_length // eps)

                log_trial(
                    filename,
                    input_size,
                    path_length,
                    eps,
                    work_units,
                    n_samples,
                    burnin,
                    width,
                    stage,
                    acceptance_rate,
                    MAP_accuracy,
                    majority_accuracy,
                    average_accuracy
                )

acceptance_rate_search("unnormalized.csv")
normalized_search("normalized.csv")