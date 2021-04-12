import numpy as np
import modelInterpreter as mi
from initialiser import get_network_initialisation_parameters

import pickle as pk

with open('../wpbc.sk.p','rb') as f:
    dummy = pk.load(f)

M = mi.modelInterpreter(dummy, 'bart')


init_parameters = get_network_initialisation_parameters(dummy, tree_model='randomforest',strength01=1e7, strength12=1e7)


HL1N, HL2N = init_parameters[2].shape

#np.save('wpbc.X.npy',X)
#np.save('wpbc.Y.npy',Y)
X = np.load('wpbc.X.npy')
Y = np.load('wpbc.Y.npy')
data = (X,X,X,Y,Y,Y) # just want to see if this runs

from k_feedforward import define_forward_pass
model = define_forward_pass(init_parameters, X.shape[1], HL1N, HL2N, n_layers=2)

from k_feedforward import run_neural_net
method2_full,_ = run_neural_net(data, init_parameters, 
                    HL1N=HL1N, HL2N=HL2N,
                    verbose=True, forest=dummy, keep_sparse=False)
method2_sparse,_ = run_neural_net(data, init_parameters, 
                    HL1N=HL1N, HL2N=HL2N,
                    verbose=True, forest=dummy, keep_sparse=True)
method1_full,_ = run_neural_net(data, init_parameters, 
                    HL1N=HL1N, HL2N=HL2N, n_layers=1,
                    verbose=True, forest=dummy, keep_sparse=False)
method3_full,_ = run_neural_net(data, init_parameters, 
                    HL1N=HL1N, HL2N=HL2N, n_layers=3,
                    verbose=True, forest=dummy, keep_sparse=False)

#RuntimeError: tf.placeholder() is not compatible with eager execution.
# need to sort this out.  Might be easier to reroll with keras
