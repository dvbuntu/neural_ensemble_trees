import numpy as np
import sys
from dataloader import load_data
from forest_fitting import fit_random_forest
from bart import fit_bart
import warnings
try:
    from lgb_fitting import TrainGBDT
except ImportError:
    warnings.warn("Problem import LightGBM")
try:
    from tqdm import tqdm
except ImportError:
    warnings.warn("No progress bar avaialble")
    tqdm = lambda x: x
from feedforward import run_neural_net
from initialiser import get_network_initialisation_parameters
from individually_trained import individually_trained_networks

# set seed and printoptions.
np.random.seed(44)
np.set_printoptions(precision=4, suppress=True)


def neural_random_forest(dataset_name="mpg", tree_model='lightgbm',
    ntrees = 150,
    depth = 6,
    tree_lr = 0.15,
    maxleaf = 100,
    mindata = 40,
    power = 2,
    base = .95,
    verbose=False,
    strength01=100,
    strength12=1,
    ):
    """
    Takes a regression dataset name, and trains/evaluates 4 classifiers:
    - a random forest
    - a 2-layer MLP
    - a neural random forest (method 1)
    - a neural random forest (method 2)
    """
    # pick a regression dataset
    dataset_names = ["boston", "concrete", "crimes", "fires", "mpg", "wisconsin", "protein", "yahoo"]
    if not dataset_name or dataset_name not in dataset_names:
        dataset_name = "wisconsin"  # set as default dataset

    # load the dataset, with randomised train/dev/test split
    data = load_data(dataset_name, seed=np.random.randint(0,100000,10)[0])

    # X: regression input variable matrix, size [n_data_points, n_features]
    # Y: regression output vector, size [n_data_points]
    # General format of data: 6-tuple
    # XTrain, XValid, XTest, YTrain, YValid, YTest

    # train a random regression forest model
    if tree_model == 'randomforest':
        model, model_results = fit_random_forest(data, ntrees, depth, verbose=verbose)
    elif tree_model == 'bart':
        model, model_results = fit_bart(data, ntrees, verbose=verbose, power=power, base=base)
    else:
        model, model_results = TrainGBDT(data, lr=tree_lr, num_trees=ntrees, maxleaf=maxleaf, mindata=mindata)

    # derive initial neural network parameters from the trained trees model
    init_parameters = get_network_initialisation_parameters(model, tree_model=tree_model, verbose=verbose, strength01=strength01, strength12=strength12)

    # determine layer size for layers 1 and 2 in the 2-layer MLP
    HL1N, HL2N = init_parameters[2].shape

    # train a standard 2-layer MLP with HL1N / HL2N hidden neurons in layer 1 / 2.
    NN2,M1 = run_neural_net(data, init_parameters=None, learning_rate=tree_lr, HL1N=HL1N, HL2N=HL2N, verbose=verbose)

    # # train many small networks individually, initial weights from a decision tree (method 1)
    # method1_full,_  = individually_trained_networks(data, ntrees, depth, keep_sparse=False, verbose=False, tree_model=tree_model)
    # method1_sparse,_ = individually_trained_networks(data, ntrees, depth, keep_sparse=True, verbose=False, tree_model=tree_model)

    # train one large network with sparse initial weights from random forest parameters (method 2)
    method2_full,M2 = run_neural_net(data, init_parameters, verbose=verbose, forest=model, keep_sparse=False, HL1N=HL1N, HL2N=HL2N)
    method2_sparse,M3 = run_neural_net(data, init_parameters, verbose=verbose, forest=model, keep_sparse=True, HL1N=HL1N, HL2N=HL2N)
    method2_sparse_fresh,M4 = run_neural_net(data, init_parameters, verbose=verbose, forest=model, keep_sparse=True, HL1N=HL1N, HL2N=HL2N, use_weights=False)

    results = {
        tree_model: model_results[2],
        "NN2": NN2,
        # "NRF1 full": method1_full,
        # "NRF1 sparse": method1_sparse,
        "NRF2 full": method2_full,
        "NRF2 sparse": method2_sparse,
        "NRF2 sparse no weights": method2_sparse_fresh,
        }

    if verbose:
        print("RMSE:", results)
    return results, model



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Try various neural network extractions of classic algorithms')
    parser.add_argument('-m','--method', action='append',
                            default=[],
                            help='Add this method to try (randomforest, bart, etc)')
    parser.add_argument('-s','--dataset', action='append',
                            default=[],
                            help='Add this data (wisconsin, etc)')
    parser.add_argument('-n','--ntrees', default=150, type=int, help='Use this many trees')
    parser.add_argument('-p','--power', default=2, type=float, help='BART power parameter')
    parser.add_argument('-b','--base', default=.95, type=float, help='BART base parameter')
    parser.add_argument('-v','--verbose', default=False, action="store_true", help='Verbose flag')
    parser.add_argument('-d','--depth', default=6, type=int, help='RF max depth')
    parser.add_argument('-l','--tree_lr', default=0.15, type=float, help='RF learning rate')
    parser.add_argument('--maxleaf', default=100, type=int, help='RF max number of leaves')
    parser.add_argument('--mindata', default=40, type=int, help='RF min data required')
    parser.add_argument('--seed', default=44, type=int, help='Random seed')
    parser.add_argument('--strength01', default=100, type=float, help='Splitting node NN conversion strength')
    parser.add_argument('--strength12', default=1, type=float, help='Leaf node NN conversion strength')
    args = parser.parse_args()
    if not args.method:
        args.method=['randomforest']
    args.method = set(args.method)
    if not args.dataset:
        args.dataset=['wisconsin']
    args.dataset = set(args.dataset)
    res = dict()
    for dataset in tqdm(args.dataset):
        for tree_model in tqdm(args.method):
            ans, model = neural_random_forest(dataset, tree_model,
                    args.ntrees,
                    args.depth,
                    args.tree_lr,
                    args.maxleaf,
                    args.mindata,
                    args.power,
                    args.base,
                    args.verbose,
                    args.strength01,
                    args.strength12)
            res[(dataset, tree_model)] = ans
    if args.verbose:
        print(res)
