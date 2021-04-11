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
    return tf.math.multiply(w,self.adj_matrix)

  def get_config(self):
    return {'adj_matrix': self.adj_matrix}

def define_forward_pass(init_parameters, n_inputs, HL1N, HL2N, sigma=1.0,
                        keep_sparse=True, n_layers=2, use_weights=True):
    """
    Defines a Multilayer Perceptron (MLP) model.
    Inputs:
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
        if init_parameters and keep_sparse:
            mask1 = tf.cast(W1 != 0, tf.float32)
            model = tf.keras.layers.Dense(HL1N,
                activation=tf.nn.tanh,
                kernel_constraint=SparselyConnected(mask1))(my_input)
        else:
            model = tf.keras.layers.Dense(HL1N,
                activation=tf.nn.tanh,
                )(my_input)
        if init_parameters and keep_sparse:
            mask2 = tf.cast(W2 != 0, tf.float32)
            model2 = tf.keras.layers.Dense(HL2N,
                activation=tf.nn.tanh,
                kernel_constraint=SparselyConnected(mask2))(model)
        else:
            model2 = tf.keras.layers.Dense(HL2N,
                activation=tf.nn.tanh,
                )(model)

        model2 = tf.clip_by_value(model2, -1,0) + 1 # rescale to avoid bad clipping
        predict = tf.keras.layers.Dense(1)(model2)

        model = tf.keras.Model(inputs=my_input, outputs=predict)
    #model.set_weights([W1,b1,W2,b2,W3, np.array([np.sum(W3)])])
        if use_weights and init_parameters:
            model.set_weights([W1,b1,W2,b2,2*W3, np.array([0])]) # b/c we rescale, double w3 and no need for bias term
        return model
    #model.set_weights([W1,b1,W2,b2,W3, 2*W3[-1]]) # last tree is bias
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


def run_neural_net(data, init_parameters=None, HL1N=20, HL2N=10, n_layers=2,
                   verbose=True, learning_rate=0.001, forest=None, keep_sparse=True,
                   batchsize=32, n_iterations=30, debug=False):
    """
    Trains / evaluates a Multilayer perceptron (MLP), potentially with a prespecified
    weight matrix initialisation.
    Inputs:
    - data: tuple of input (X) - output (Y) data for train/dev/test set.
        Output of dataloader.split_data
    - init_parameters: output of initialiser.get_network_initialisation_parameters.
        if init_parameters is set to None, random initial weights are picked.
    - HL1N: number of neurons in first hidden layer
    - HL2N: number of neurons in second hidden layer
    - n_layers: number of hidden layers. Default 2, but can also be used with 1 and 3.
    - verbose: how much to print
    - learning_rate: used during training
    - forest: a pre-trained random forest model. Not relevant when random initialisation is used.
    - keep_sparse: whether to enforce weight matrix sparsity during training
    - batchsize: used during training
    - n_iterations: Number of training epochs
    """

    if debug:
        import pdb
        pdb.set_trace()
    if verbose:
        print("training MLP...")
    XTrain, XValid, XTest, YTrain, YValid, YTest = data
    n_samples, n_inputs = XTrain.shape
    batchsize = min(batchsize, n_samples)
    #ops.reset_default_graph()


    # forward pass
    model = define_forward_pass(init_parameters, n_inputs, HL1N, HL2N, n_layers=n_layers)
    #prediction = lambda X: define_forward_pass(X, init_parameters, n_inputs, HL1N, HL2N, n_layers=n_layers)

    # defining a RMSE objective function
    #loss = tf.reduce_mean(input_tensor=tf.pow(prediction - Y, 2) )
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    #optimiser = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)
    model.compile(loss='mse', optimizer=optimizer)

    # define minibatch boundaries
    batch_boundaries = list(zip(range(0, n_samples, batchsize), \
                range(batchsize, n_samples, batchsize)))
    if n_samples % batchsize:
        batch_boundaries += [(batch_boundaries[-1][1],n_samples)]
    if len(batch_boundaries) == 0:
        batch_boundaries += [(0,n_samples)]


    RMSE_train, RMSE_valid, RMSE_test = [], [], []
    pred_test_store = []

    for i in range(n_iterations):
        #shuffle training data new in every epoch
        perm = np.random.permutation(n_samples)
        XTrain = XTrain[perm,:]
        YTrain = YTrain[perm]


        ## do (possibly additional) training
        model.fit(XTrain, YTrain, batch_size=batchsize)

        pred_train = model.predict(XTrain)
        pred_test = model.predict(XTest)
        pred_valid = model.predict(XValid)
        #pred_train = EvalTestset(sess, X, Y, XTrain, YTrain, model)
        #pred_test = EvalTestset(sess, X, Y, XTest, YTest, model)
        # pred_train = sess.run(prediction, feed_dict={X: XTrain, Y: YTrain})
        # pred_valid = sess.run(prediction, feed_dict={X: XValid[:128], Y: YValid[:128]})
        # pred_test = sess.run(prediction, feed_dict={X: XTest, Y: YTest})
        pred_test_store.append(pred_test)

        diff_train = YTrain - pred_train
        RMSE_train.append(np.mean(np.square(diff_train ) ) )

        diff_valid = YValid - pred_valid
        RMSE_valid.append(np.mean(np.square(diff_valid ) ) ) 

        diff_test = YTest - pred_test
        RMSE_test.append( np.mean(np.square(diff_test ) ) ) 
        if verbose:
            printstring = "Epoch: {}, Train/Test RMSE: {}"\
                    .format(i, np.array([RMSE_train[-1], RMSE_test[-1]]))
            print (printstring)


    # minimum validation error
    amin = np.argmin(RMSE_valid)
    if verbose:
        print ("argmin at", amin )
        print ("valid:", RMSE_valid[amin] )
        print ("test:", RMSE_test[amin] )


    if forest is None:  # vanilla neural net
        return RMSE_test[amin], pred_test_store[amin]
    else:               # RF-initialised neural net

        # In some cases, the tuned RF is not better than the original RF.
        # Validation accuracy is used to identify these cases.

        # compute RF validation performance
        RF_predictions_valid = forest.predict(XValid)
        RF_score_valid = np.mean (np.square(RF_predictions_valid-np.squeeze(YValid) ) )

        # if RF validation performance is better than for neural model
        if RF_score_valid < RMSE_valid[amin]:
            # Case Yes -- return forest score / predictions
            RF_predictions_test = forest.predict(XTest)
            RF_score_test =  np.mean (np.square(RF_predictions_test-np.squeeze(YTest) ) ) 
            return RF_score_test, RF_predictions_test
        else:
            # Case No -- return tuned model score / predictions
            return RMSE_test[amin], pred_test_store[amin]


