import tensorflow as tf
import abc
from tensorflow.keras.activations import softmax
from tensorflow.keras.losses import categorical_crossentropy

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
        self.nl_prior = nl_prior / self.var

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