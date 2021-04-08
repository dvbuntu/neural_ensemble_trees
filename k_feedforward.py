import numpy as np
import tensorflow as tf
import math

'''
Keras-based feedforward module
'''

## use weight constraint to enforce sparse connections, not efficient...
class SparselyConnected(tf.keras.constraints.Constraint):
  """Constrains weight tensors to be sparsely connected according to `adj_matrix`"""

  def __init__(self, adj_matrix):
    self.adj_matrix = adj_matrix

  def __call__(self, w):
    return tf.math.multiply(w,adj_matrix)

  def get_config(self):
    return {'adj_matrix': self.adj_matrix}

def define_forward_pass(X, init_parameters, n_inputs, HL1N, HL2N, sigma=1.0,
                        keep_sparse=True, n_layers=2):
    """
    Defines a Multilayer Perceptron (MLP) model.
    Inputs:
        - X: tf placeholder [batchsize, n_inputs] for feature inputs
        - init_parameters: tuple of numpy arrays (or None), holding values to initialise
                the network weight matrices (and biases). If None, random values are
                used for network initialisation.
        - n_inputs: number of features
        - HL1N: hidden layer 1 size
        - Hl2N: hidden layer 2 size
        - sigma: Variance of Gaussian distribution used when randomly initalising weights
        - keep_sparse: whether to enforce network weight sparsity
        - n_layers: number of hidden layers. Default case is 2, but can also run with 1 or 3,
            but the number of neurons is then defined by HL1N, HL2N.
            Note: The RF initialisation can only be used with n_layers = 2.
    Outputs:
        - predictions: tf tensor holding model predictions
    """

    if init_parameters:
        W1, b1, W2, b2, W3 = init_parameters

    if n_layers == 2:   # default case.
        my_input = tf.keras.Input(shape=(n_inputs,))
        if init_parameters:
            mask1 = tf.cast(W1 != 0, tf.float32)
	    model = tf.keras.layers.Dense(HL1N,
			 activation=tf.nn.tanh,
			 kernel_constraint=SparselyConnected(mask1))(my_input)
        if init_parameters:
            mask2 = tf.cast(W2 != 0, tf.float32)
	    model2 = tf.keras.layers.Dense(HL2N,
			 activation=tf.nn.tanh,
			 kernel_constraint=SparselyConnected(mask2))(model)
            model2 = tf.clip_by_value(model2, -1,0) + 1 # rescale
        predict = tf.keras.layers.Dense(1)(model2)

        model = tf.keras.Model(inputs=my_input, outputs=predict)
	model.set_weights([W1,b1,W2,b2,W3, np.array([np.sum(W3)])])
	model.set_weights([W1,b1,W2,b2,2*W3, np.array([0])])
	model.set_weights([W1,b1,W2,b2,W3, 2*W3[-1]]) # last tree is bias
	# getting highly correlated but not identical values to actual forest
        # off by a factor of 2 in the W3 values?  But tanh is +-1?
        # do we really want a sigmoid here?

        # need a bigger strength param perhaps?
        # that might be it, try very big, 100000
        # and fix bias term, increase strenght12

        # two results are off...from the sklearn forest, same two that it was wrong?
        # all weirdly biased, issue is the tanh kills off a few terms
        # getting zeros in output of first and 2nd layer
        # first layer not an issue, but second layer is problem

