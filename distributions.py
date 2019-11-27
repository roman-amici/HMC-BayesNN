import tensorflow as tf
import abc
from tensorflow.keras.activations import softmax
from tensorflow.keras.losses import categorical_crossentropy
import numpy as np

class Distribution(abc.ABC):

    @abc.abstractmethod
    def negative_log_posterior(self,q):
        pass

    @abc.abstractmethod
    def negative_log_posterior_gradient(self,q):
        pass

    @abc.abstractmethod
    def probs(self,X,q):
        pass

class Bayesian_NN_Eager(Distribution):

    def __init__(self, X_train, y_train, var=2.0):

        assert(tf.executing_eagerly())

        self.X_train = tf.constant(X_train, dtype=tf.float32)
        self.y_train = tf.constant(y_train, dtype=tf.float32)

        self.var = var

    @staticmethod
    def to_tensor(numpy_list):
        return [ tf.convert_to_tensor for l in numpy_list ]

    def negative_log_posterior(self,WS):
        WS = self.to_tensor(WS)

        nl_prior = 0
        for w in WS:
            nl_prior += tf.reduce_sum(w**2)
        nl_prior /= 2 * self.var

        Z = self.X_train
        for w in WS[:-1]:
            Z = tf.tanh(Z @ w)
        y_hat = softmax(Z @ WS[-1])

        nll = tf.reduce_sum(categorical_crossentropy(self.y_train,y_hat))

        return nll + nl_prior

    def negative_log_posterior_gradient(self,WS):
        WS = self.to_tensor(WS)

        with tf.GradientTape() as g:
            g.watch(WS)

            nlp = self.negative_log_posterior(WS)

        return g.gradient(nlp,WS)

    def probs(self, X_test, WS):
        Z = self.X_train
        for w in WS[:-1]:
            Z = tf.tanh(Z @ w)
        return softmax(Z @ WS[-1])

class Bayesian_NN_Session(Distribution):

    def __init__(self, X_train, y_train, layers, var=2.0):
        self.X_train = tf.constant(X_train, dtype=tf.float32)
        self.y_train = tf.constant(y_train, dtype=tf.float32)
        self.layers = layers
        self.var = var

        self.sess = tf.Session()

        self.weights = tuple([tf.placeholder(tf.float32, shape=dim) for dim in layers ])

        nl_prior = 0
        for w in self.weights:
            nl_prior += tf.reduce_sum(w**2)
        self.nl_prior = nl_prior / (2*self.var)

        Z = self.X_train
        for w in self.weights[:-1]:
            Z = tf.tanh(Z @ w)
        self.y_hat_train = softmax(Z @ self.weights[-1])
        self.nll = tf.reduce_sum(categorical_crossentropy(self.y_train, self.y_hat_train))

        self.loss = self.nll + self.nl_prior

        self.grads = [ tf.gradients(self.loss, w)[0] for w in self.weights]
        
        self.X_test = tf.placeholder(tf.float32, shape=(None,layers[0][0]))
        
        Z = self.X_test
        for w in self.weights[:-1]:
            Z = tf.tanh(Z @ w)
        self.y_hat = softmax(Z @ self.weights[-1])

    def negative_log_posterior(self,WS):
        return self.sess.run(self.loss, {self.weights : tuple(WS) })

    def negative_log_posterior_gradient(self,WS):
        return self.sess.run(self.grads, {self.weights : tuple(WS) })

    def probs(self,X_test,WS):
        return self.sess.run(self.y_hat, {self.weights: tuple(WS), self.X_test : X_test})

class Bayesian_NN_Session_2Layer(Distribution):
    def __init__(self, X_train, y_train, layers, var=2.0):
        self.X_train = tf.constant(X_train, dtype=tf.float32)
        self.y_train = tf.constant(y_train, dtype=tf.float32)
        self.layers = layers
        self.var = var

        self.sess = tf.Session()

        self.weights = tuple([tf.placeholder(tf.float32, shape=dim) for dim in layers ])
        
        self.X_test = 1
        Z = self.X_test
        for w in self.weights[:-1]:
            Z = tf.tanh(Z @ w)
        self.y_hat = softmax(Z @ self.weights[-1])

def __negative_log_posterior(self,X,y,ws):
        nl_prior = 0
        for w in ws:
            nl_prior += tf.reduce_sum(w**2)
        nl_prior = nl_prior / (2*self.var)

        Z = X
        for w in ws:
            Z = tf.tanh(Z @ w)
        y_hat = softmax(Z @ self.weights[-1])
        nll = tf.reduce_sum(categorical_crossentropy(y, y_hat))

        return nll + nl_prior

def __negative_log_posterior_gradient(self, X, y, ws):
    loss = self.__negative_log_posterior(X,y,ws)
    grads = [ tf.gradients(loss, w)[0] for w in ws]
    return grads

def __leapfrog(self, ):
    pass

def negative_log_posterior(self,WS):
    return self.sess.run(self.loss, {self.weights : tuple(WS) })

def negative_log_posterior_gradient(self,WS):
    return self.sess.run(self.grads, {self.weights : tuple(WS) })

def probs(self,X_test,WS):
    return self.sess.run(self.y_hat, {self.weights: tuple(WS), self.X_test : X_test})

class Stochastic_Bayesian_NN_Session(Distribution):

    def __init__(self, input_dim, output_dim, layers, var=2.0):
        self.X = tf.placeholder(tf.float32, shape=[None, input_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, output_dim])
        self.layers = layers
        self.var = var

        self.sess = tf.Session()

        weights_total = np.sum([d1*d2 for d1,d2 in layers])

        self.weights_input = tf.placeholder(tf.float32, shape=[weights_total,1])

        #Wrap the input weights vector
        self.weights = []
        start = 0
        for d1,d2 in layers:
            end = start + d1*d2
            seq = self.weights_input[:,start:end]
            self.weights.append( tf.reshape( seq, (d1,d2) ) )
            start = end

        nl_prior = tf.reduce_sum(self.weights_input**2)
        self.nl_prior = nl_prior / (2*self.var)

        self.y_hat = self.__NN()
        self.log_prob = categorical_crossentropy(self.y, self.y_hat)
        self.nll = tf.reduce_sum(self.log_prob)
        self.loss = self.nll + self.nl_prior

        self.grad = tf.gradients(self.loss, self.weights_input)[0]

        self.emperical_fisher = self.__emperical_fisher()

    
    def __NN(self):
        Z = self.X
        for w in self.weights[:-1]:
            Z = tf.tanh(Z @ w)
        return softmax(Z @ self.weights[-1])

    def __emperical_fisher(self):
        g_expansion = tf.gradients(self.log_prob, self.weights_input)[0]
        g_bar = tf.reduce_mean(g_expansion,axis=0,keep_dims=True)

        return g_expansion - g_bar

    def set_train(self,X_train, y_train):
        self.X_train_feed = X_train
        self.y_train_feed = y_train

    def negative_log_posterior(self,w):
        feed_dict = {self.X: self.X_train_feed, self.y : self.y_train_feed, self.weights_input : w }
        return self.sess.run(self.loss, feed_dict)

    def negative_log_posterior_gradient(self,w):
        feed_dict = {self.X : self.X_train_feed, self.y : self.y_train_feed, self.weights_input : w }
        return self.sess.run(self.grad, feed_dict)

    def probs(self,X_test,w):
        return self.sess.run(self.y_hat, {self.weights_input: w, self.X : X_test})

    def emperical_fisher_information(self,w):
        unnormalized = self.sess.run(self.emperical_fisher, {self.weights_input : w, self.X : self.X_train_feed, self.y : self.y_train_feed})
        return unnormalized / (self.X_train_feed.shape[0] - 1)