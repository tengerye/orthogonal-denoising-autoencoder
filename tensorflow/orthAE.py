import tensorflow as tf
import numpy as np
from random import randint



class AE(object):
    """The class implements the basic auto-encoder."""
    
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer = tf.train.AdamOptimizer()):
        """The constructor of an auto-encoder.

        Args:
        n_input: `int`, the dimension of input layer.
        n_hidden: `int`, the dimension of hidden layer.
        transfer_function: `Operation`, activation function.
        optimizer: `Operation`, optimizer to train the model.
        """
        
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function

        self.weights = self._initialize_weights()

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)


    def _initialize_weights(self):
        """Initialize the auto-encoder as soon as we created it.
        
        The initialization includes w1, b1, w2, b2.
        
        Returns:
            all_weights: the dictionary containing initialized variables.
        """
        all_weights = dict()
        
        all_weights['w1'] = tf.get_variable("w1", shape=[self.n_input, self.n_hidden])
        all_weights['b1'] = tf.get_variable("b1", shape=[self.n_hidden])
        all_weights['w2'] = tf.get_variable("w2", shape=[self.n_hidden, self.n_input])
        all_weights['b2'] = tf.get_variable("b2", shape=[self.n_input])
        
        return all_weights
    

    def train(self, X):
        cost, _ = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {self.x: X})

    
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X})

    
    def generate(self, hidden = None):
        if hidden is None:
            hidden = self.sess.run(tf.random_normal([1, self.n_hidden]))
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    
    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X})

    
    def getBiases(self):
        return self.sess.run(self.weights['b1'])
    
    def getWeights(self):
        return self.sess.run(self.weights)
    
    
class OrthAE(AE):
    """The class implements orthogonal auto-encoder."""
    
    def __init__(self, array_n_input, array_n_hidden, transfer_function=tf.nn.softplus, 
                 optimizer = tf.train.GradientDescentOptimizer(0.5)):
        """Initialization.
        
        Args:
        array_n_input: an array or list indicating the number of dimensions of each view.
        array_n_hidden: an array or list indicating the number of dimensions of spaces, 
        the last one corresponds to private space.
        transfer_function: activation function.
        """
        self.array_n_input = array_n_input
        self.array_n_hidden = array_n_hidden
        self.transfer = transfer_function
        
        # Lengths of input layer and hidden layer.
        self.n_input = np.sum(array_n_input)
        self.n_hidden = np.sum(array_n_hidden)

        self.weights = self._initialize_weights()
        
        # model
        self.x = tf.placeholder(tf.float32, shape=[None, self.n_input])

        self.hidden = self.transfer(tf.add(tf.matmul(self.x, 
                                                     tf.multiply(self.weights['w1'], self.weights['mask_w1'])), 
                                           tf.multiply(self.weights['b1'], self.weights['mask_b1'])))
        
        self.reconstruction = tf.add(tf.matmul(self.hidden, 
                                               tf.multiply(self.weights['w2'], self.weights['mask_w2'])),
                                     self.weights['b2'])
                
        # cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        
        
    def _initialize_weights(self):
        """Initialize the auto-encoder as soon as we created it.
        
        The initialization includes: w1, b1, w2, b2, mask_w1, mask_b1.
        
        Returns:
            all_weights: the dictionary containing initialized variables.
        """
        
        # w1, b1, w2, b2
        all_weights = AE._initialize_weights(self)
        
        # Instead of setting disconnections to zeros, we can mask the weights.
        
        # mask_w1
        mask_w1 = np.zeros((self.n_input, self.n_hidden)).astype(np.float32)
        
        array_n_input = self.array_n_input
        array_n_hidden = self.array_n_hidden
        
        mask_w1[:, 0 : array_n_hidden[0]] = 1
        input_idx, hidden_idx = 0, array_n_hidden[0]
        
        for n_idx in range(len(array_n_input)):
            mask_w1[input_idx : input_idx+array_n_input[n_idx], hidden_idx : hidden_idx+array_n_hidden[n_idx+1]] = 1
            input_idx, hidden_idx = input_idx+array_n_input[n_idx], hidden_idx+array_n_hidden[n_idx+1]
        
        all_weights['mask_w1'] = tf.constant(mask_w1)
        
        # mask_b1
        mask_b1 = np.zeros((self.n_hidden)).astype(np.float32)
        mask_b1[:array_n_hidden[1]] = 1
        
        all_weights['mask_b1'] = tf.constant(mask_b1)
        
        # mask_w2
        all_weights['mask_w2'] = tf.constant(np.transpose(mask_w1))
        
        return all_weights
    
    
    def getWeights(self):
        """Overwrite the mothod from AE to return masked weights."""
        all_weights = dict()
        
        w1, mask_w1, b1, mask_b1, w2, mask_w2, b2 = self.sess.run(
            [self.weights['w1'], self.weights['mask_w1'], self.weights['b1'], self.weights['mask_b1'],
             self.weights['w2'], self.weights['mask_w2'], self.weights['b2']])
        
        all_weights['w1'], all_weights['b1'], all_weights['w2'] = \
        np.multiply(w1, mask_w1), np.multiply(b1, mask_b1), np.multiply(w2, mask_w2)
            
        all_weights['b2'] = b2
        return all_weights
        
        
class OrthdAE(OrthAE):
    """The class implements orthogonal denoising auto-encoder."""
    
    def __init__(self, array_n_input, array_n_hidden, transfer_function=tf.nn.softplus, 
                 optimizer = tf.train.AdamOptimizer()):
        """Initialization. y is the noising version of x. Both x and y are placeholders
        to be fed.
        
        Args:
        array_n_input: an array or list indicating the number of dimensions of each view.
        array_n_hidden: an array or list indicating the number of dimensions of spaces, 
        the last one corresponds to private space.
        transfer_function: activation function.
        x: original data.
        y: corrupted version of x (same size of x).
        """
        self.array_n_input = array_n_input
        self.array_n_hidden = array_n_hidden
        self.transfer = transfer_function
        
        # Lengths of input layer and hidden layer.
        self.n_input = np.sum(array_n_input)
        self.n_hidden = np.sum(array_n_hidden)

        self.weights = self._initialize_weights()
        
        # model
        self.x = tf.placeholder(tf.float32, shape=[None, self.n_input])
        self.y = tf.placeholder(tf.float32, shape=[None, self.n_input])

        self.hidden = self.transfer(tf.add(tf.matmul(self.y, 
                                                     tf.multiply(self.weights['w1'], self.weights['mask_w1'])), 
                                           tf.multiply(self.weights['b1'], self.weights['mask_b1'])))
        
        self.reconstruction = tf.add(tf.matmul(self.hidden, 
                                               tf.multiply(self.weights['w2'], self.weights['mask_w2'])),
                                     self.weights['b2'])
                
        # cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        

    def train(self, X, Y = None, max_iter = 1000000):
        """Based on the original data (X) and the corrupted data (Y), we train the model."""
        if Y is None:
            Y = trivial_denoising(X)
            
        # Itervatively train.
        for i in range(max_iter):
            cost, _ = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X, self.y: Y})
            
            if i % 10000 == 0:
                print('Step: ' + str(i), ': cost =', cost)
        return cost
    
    def transform(self, X):
        """Override."""
        return self.sess.run(self.hidden, feed_dict={self.y: X})
    

def trivial_denoising(X):
    """Trivial denoising method. A random feature is whitening.
    
    Args:
    X: original data. Its shape is [n_samples, n_features].
    
    Returns:
    Y: denoised data. Its shape is [n_samples, n_features].
    """
#     n, m = X.shape
#     Y = np.tile(X, [m, 1])

#     for i in range(m):
#         Y[i*n : i*n+n, i] = 0

    n, m = X.shape
    Y = X[:]

    Y[:, randint(0, m)] = 0
        
    return Y
