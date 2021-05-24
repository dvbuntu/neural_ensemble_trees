import numpy as np
import tensorflow as tf
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

'''
Keras-based feedforward module
'''

checkpoint_filepath = '/tmp/checkpoint'

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
                        keep_sparse=True, n_layers=2, use_weights=True,
                        regularize=0.1, reg_type=tf.keras.regularizers.l2):
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
        - model: keras model to create predictions
    """

    if init_parameters:
        W1, b1, W2, b2, W3 = init_parameters

    my_input = tf.keras.Input(shape=(n_inputs,))
    if n_layers == 2:   # default case.
        if init_parameters and keep_sparse:
            mask1 = tf.cast(W1 != 0, tf.float32)
            model = tf.keras.layers.Dense(HL1N,
                activation=tf.nn.tanh,
                kernel_regularizer=reg_type(regularize),
                kernel_constraint=SparselyConnected(mask1))(my_input)
        else:
            model = tf.keras.layers.Dense(HL1N,
                activation=tf.nn.tanh,
                kernel_regularizer=reg_type(regularize),
                )(my_input)
        if init_parameters and keep_sparse:
            mask2 = tf.cast(W2 != 0, tf.float32)
            model2 = tf.keras.layers.Dense(HL2N,
                activation=tf.nn.tanh,
                kernel_regularizer=reg_type(regularize),
                kernel_constraint=SparselyConnected(mask2))(model)
        else:
            model2 = tf.keras.layers.Dense(HL2N,
                activation=tf.nn.tanh,
                kernel_regularizer=reg_type(regularize),
                )(model)

        # orchards use sigmoid
        if init_parameters:
            model2 = tf.clip_by_value(model2, -1,0) + 1 # rescale to avoid bad clipping
        predict = tf.keras.layers.Dense(1,
                        kernel_regularizer=reg_type(regularize),
                    )(model2)

        model = tf.keras.Model(inputs=my_input, outputs=predict)
    #model.set_weights([W1,b1,W2,b2,W3, np.array([np.sum(W3)])])
        if use_weights and init_parameters:
            model.set_weights([W1,b1,W2,b2,2*W3, np.array([0])]) # b/c we rescale, double w3 and no need for bias term
    # standard 1-layer MLP
    elif n_layers == 1:
        model = tf.keras.layers.Dense(HL1N,
                kernel_regularizer=reg_type(regularize),
                activation=tf.nn.tanh)(my_input)
        predict = tf.keras.layers.Dense(1,
                        kernel_regularizer=reg_type(regularize),
                    )(model)
        model = tf.keras.Model(inputs=my_input, outputs=predict)
    # fully connect 3-layer MLP of same size as forest
    elif n_layers == 3:
        model = tf.keras.layers.Dense(HL1N,
                kernel_regularizer=reg_type(regularize),
                activation=tf.nn.tanh)(my_input)
        model2 = tf.keras.layers.Dense(HL2N,
                    kernel_regularizer=reg_type(regularize),
                    activation=tf.nn.tanh,
                )(model)
        predict = tf.keras.layers.Dense(1,
                        kernel_regularizer=reg_type(regularize),
                    )(model2)
        model = tf.keras.Model(inputs=my_input, outputs=predict)
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
                   batchsize=32, n_iterations=30, debug=False, use_weights=True,
                   regularize=0.1, reg_type=tf.keras.regularizers.l2, seed=42):
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
    tf.random.set_seed(seed)
    XTrain, XValid, XTest, YTrain, YValid, YTest = data
    n_samples, n_inputs = XTrain.shape
    batchsize = min(batchsize, n_samples)
    #ops.reset_default_graph()

    if isinstance(reg_type,str):
        if reg_type.lower().startswith('l1'):
            reg_type = tf.keras.regularizers.l1
        elif reg_type.lower().startswith('l2'):
            reg_type = tf.keras.regularizers.l2


    # forward pass
    model = define_forward_pass(init_parameters, n_inputs, HL1N, HL2N, n_layers=n_layers, use_weights=use_weights, regularize=regularize, reg_type=reg_type)
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

    # much faster than iterating model.fit
    ## update this to do checkpointing
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        )

    # save a model prior to training
    model.save_weights(checkpoint_filepath+'_init')
    pred_valid = model.predict(XValid)
    rmse_v = np.sqrt(np.mean(np.square(np.squeeze(YValid) - np.squeeze(pred_valid) ) ) )
    rmse_t = np.sqrt(np.mean(np.square(np.squeeze(YTest) - np.squeeze(model.predict(XTest)) ) ) )

    model.fit(XTrain, YTrain, batch_size=batchsize, verbose=verbose,
            epochs=n_iterations, shuffle=True,
            callbacks=[model_checkpoint_callback],
            validation_data=(XValid,YValid))
    # restore best weights
    model.load_weights(checkpoint_filepath)
    # check weights prior to training
    pred_valid = model.predict(XValid)

    # reset to better original weights
    if np.sqrt(np.mean(np.square(np.squeeze(YValid) - np.squeeze(pred_valid) ) ) ) > rmse_v:
        model.load_weights(checkpoint_filepath+'_init')

    pred_train = model.predict(XTrain)
    pred_valid = model.predict(XValid)
    pred_test = model.predict(XTest)
    RMSE_train.append(np.sqrt(np.mean(np.square(np.squeeze(YTrain) - np.squeeze(pred_train) ) ) ))
    RMSE_valid.append(np.sqrt(np.mean(np.square(np.squeeze(YValid) - np.squeeze(pred_valid) ) ) ))
    RMSE_test.append(np.sqrt(np.mean(np.square(np.squeeze(YTest) - np.squeeze(pred_test) ) ) ))
    pred_test_store.append(pred_test)

    # for debugging
    #if init_parameters and use_weights:
    #    print(rmse_v, RMSE_valid[0], rmse_t, RMSE_test[0])
    #    raise ValueError

    if False:
        for i in range(n_iterations+1):
            #shuffle training data new in every epoch
            perm = np.random.permutation(n_samples)
            XTrain = XTrain[perm,:]
            YTrain = YTrain[perm]


            ## do (possibly additional) training
            ## first iteration is just plain results w/o extra training
            if i:
                model.fit(XTrain, YTrain, epochs=1, batch_size=batchsize, verbose=verbose)

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
            RMSE_train.append(np.sqrt(np.mean(np.square(diff_train ) ) ))

            diff_valid = YValid - pred_valid
            RMSE_valid.append(np.sqrt(np.mean(np.square(diff_valid ) ) ) )

            diff_test = YTest - pred_test
            RMSE_test.append(np.sqrt( np.mean(np.square(diff_test ) ) ) )
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


    #if forest is None:  # vanilla neural net
    ## don't ever want the old model
    if True:
        return RMSE_test[amin], pred_test_store[amin]
    else:               # RF-initialised neural net

        # In some cases, the tuned RF is not better than the original RF.
        # Validation accuracy is used to identify these cases.

        # compute RF validation performance
        RF_predictions_valid = forest.predict(XValid)
        RF_score_valid = np.sqrt(np.mean (np.square(RF_predictions_valid-np.squeeze(YValid) ) ))

        # if RF validation performance is better than for neural model
        if RF_score_valid < RMSE_valid[amin]:
            # Case Yes -- return forest score / predictions
            RF_predictions_test = forest.predict(XTest)
            RF_score_test =  np.mean (np.square(RF_predictions_test-np.squeeze(YTest) ) ) 
            return RF_score_test, RF_predictions_test
        else:
            # Case No -- return tuned model score / predictions
            return RMSE_test[amin], pred_test_store[amin]


