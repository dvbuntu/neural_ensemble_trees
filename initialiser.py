from layer_initialisation import InitFirstLayer
from layer_initialisation import InitSecondLayer
from layer_initialisation import InitThirdLayer
from modelInterpreter import modelInterpreter

def get_network_initialisation_parameters(model, strength01=100.0, strength12=1.0, tree_model='lightgbm', verbose=True):
    """
    Given a pre-trained random forest model, this function returns as numpy arrays
    the weights and biases for initialising a 2-layer feedforward neural network.
    The strength01 and strength12 are hyperparameters that determine how strongly
    the continuous neural network nonlinearity will approximate a discrete step function
    """

    modelI = modelInterpreter(model, tree_model, verbose)

    # get network parameters for first hidden layer
    W1, b1, nodelist1 = InitFirstLayer(modelI, strength01)

    # get network parameters for second hidden layer
    W2, b2, leaf_neurons = InitSecondLayer(modelI, nodelist1, strength12)

    # get network parameters for third hidden layer
    W3 = InitThirdLayer(modelI, leaf_neurons)

    return W1, b1, W2, b2, W3

