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

from feedforward import define_forward_pass
model = define_forward_pass(init_parameters, X.shape[1], HL1N, HL2N, n_layers=2, keep_sparse=True)
W0 = model.get_weights()

import tensorflow as tf
optimizer = tf.keras.optimizers.Adam(0.015)
#optimiser = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)
model.compile(loss='mse', optimizer=optimizer)
model.fit(X, Y, batch_size=32, verbose=True, epochs=30)
W1 = model.get_weights()
diff = [w0-w1 for w0,w1 in zip(W0,W1)]
print([np.max(np.abs(d)) for d in diff]) # only changes final values slightly

model = define_forward_pass(init_parameters, X.shape[1], HL1N, HL2N, n_layers=2, keep_sparse=False)
model.compile(loss='mse', optimizer=optimizer)
model.fit(X, Y, batch_size=32, verbose=True, epochs=30)
W1 = model.get_weights()
diff = [w0-w1 for w0,w1 in zip(W0,W1)]
print([np.max(np.abs(d)) for d in diff]) # changes to a single split?


from feedforward import run_neural_net
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

##rf
# RMSE: {'randomforest': 40.22405954173736, 'NN2': 33.678523275861814, 'NRF2 full': 37.5213766559505, 'NRF2 sparse': 37.74709911047756}
## bart
# RMSE: {'bart': 35.61804841560612, 'NN2': 41.26699087356276, 'NRF2 full': 34.585775930395904, 'NRF2 sparse': 34.56567196957474}
