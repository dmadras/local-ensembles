"""Run the Local Ensembles methods on a pretrained model and some test inputs.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from local_ensembles import load_model
from local_ensembles import second_order as so
import local_ensembles.evaluation_functions as eval_fns
import local_ensembles.lanczos_functions as L
from run_baselines import run_baselines
from utils import utils
from utils import tensor_utils as tu
from utils.dataset_iterator import DatasetIterator

EPS = 1e-10
CHUNK = 10000
ORTHO_BASIS_NAME = 'lanczos_Q_lan'
BETA_NAME = 'lanczos_Beta'
ALPHA_NAME = 'lanczos_Alpha'
EVECS_NAME = 'lanczos_A_evecs_unsorted'

def get_pred_grads_in_minibatches(x, pred_fn, map_grad_fn, mb_size):
    """Gets predictions gradients for a batch of examples x.

    Args:
        x (tensor): the examples we wish to get the prediction gradients for.
        pred_fn (function): a function which returns a vector of model
            predictions for a given batch.
        map_grad_fn (function): a function which takes the gradient of each
            element of a vector.
        mb_size (int): how many examples of x to get the gradient for at once.
    Returns:
        grads (tensor): prediction gradients for each element in x. If x
            is n x d and our model has p parameters, grads is n x p.
    """
    print('Getting gradients in minibatches.')
    curr = 0
    num_examples = x.shape[0]
    grads = []
    while curr < num_examples:
        batch_x = x[curr: min(curr + mb_size, num_examples)]
        pred_grads_tf = tu.flat_concat(
            so.get_pred_grads(batch_x, pred_fn, map_grad_fn))
        pred_grads = pred_grads_tf.numpy()
        del pred_grads_tf
        grads.append(pred_grads)
        curr += mb_size
    grads = np.concatenate(grads, axis=0)
    return grads

def get_loss_grads_in_minibatches(x, y, loss_fn, map_grad_fn, mb_size):
    """Gets loss gradients for a batch of examples x.

    Args:
        x (tensor): the examples we wish to get the loss gradients for.
        y (tensor): the labels we wish to get the loss gradients for.
        loss_fn (function): a function which returns a vector of losses
            for a given batch.
        map_grad_fn (function): a function which takes the gradient of each
            element of a vector.
        mb_size (int): how many examples of x to get the gradient for at once.
    Returns:
        grads (tensor): loss gradients for each element in x. If x
            is n x d and our model has p parameters, grads is n x p.
    """
    print('Getting gradients in minibatches.')
    curr = 0
    num_examples = x.shape[0]
    grads = []
    while curr < num_examples:
        batch_x = x[curr: min(curr + mb_size, num_examples)] 
        batch_y = y[curr: min(curr + mb_size, num_examples)] 
        loss_grads_tf = tu.flat_concat(
                so.get_loss_grads(batch_x, batch_y, loss_fn, map_grad_fn))
        loss_grads = loss_grads_tf.numpy()
        del loss_grads_tf
        grads.append(loss_grads)
        curr += mb_size
    grads = np.concatenate(grads, axis=0)
    return grads

def get_model_dim(model):
    return sum([tf.size(w) for w in model.weights])

def load_saved_data(expdir, num_examples, n_batch, start_ix=0):
    """Load saved data tensors from the train/valid/test/ood sets.

    Args:
        expdir (str): path where the tensors are saved.
        num_examples (int): how many examples to return in each tensor.
        start_ix (int, optional): where to start indexing when returning data.
    Returns:
        iterators for training, validation, test, and OOD data.
    """
    phases = ['train', 'valid', 'test', 'ood']
    data_objects = ['x', 'y']
    res = {}
    for phase in phases:
        for obj in data_objects:
            fname = os.path.join(expdir, 'tensors',
                                 '{}_{}.npy'.format(phase, obj))
            saved_data = np.load(fname)
            res[(phase, obj)] = saved_data[start_ix:start_ix + num_examples]
            del saved_data

    itr_training_data = DatasetIterator(res[('train', 'x')],
                                        res[('train', 'y')], n_batch)
    itr_valid = DatasetIterator(res[('valid', 'x')],
                                res[('valid', 'y')], n_batch)
    itr_test = DatasetIterator(res[('test', 'x')],
                               res[('test', 'y')], n_batch)
    itr_ood = DatasetIterator(res[('ood', 'x')],
                              res[('ood', 'y')], n_batch)

    return itr_training_data, itr_valid, itr_test, itr_ood

def calculate_aucs(res, cutoff_list, tensordir):
    """Calculate AUCs from extrapolation scores.

    Args:
        res (dict): dictionary of calculated extrapolation scores.
        cutoff_list (list of ints): stopping values of the Lanczos iteration
            we have tried (m in the paper).
        tensordir (str): where to store extrapolation scores.
    """
    aucs = []
    for cutoff in cutoff_list:
        res_combined = {}
        for dist in ['test', 'ood']:
            # Take the min across several labels if using loss grads.
            # If using prediction grads, this operation does nothing.
            scores = np.amin(res[dist][cutoff], axis=1)
            utils.save_tensors([('{}_{:d}'
                .format(dist, cutoff),
                res[dist][cutoff])], tensordir)
            res_combined[dist] = scores
        auc = eval_fns.get_auc(
            res_combined['test'], res_combined['ood'])
        aucs.append(auc)
        print('{:d} E-vecs: AUC (min) = {:.2f}'.format(cutoff, auc))
    print('AUCs', aucs)
    return aucs

def plot_aucs(figdir, cutoff_list, aucs):
    plt.clf()
    plt.plot(cutoff_list, aucs)
    plt.savefig(os.path.join(figdir,
                             'lanczos_ood_all.pdf'))

def record_aucs(logdir, fname, item_list):
    logfile_name = os.path.join(logdir, fname)
    logfile = open(logfile_name, 'w')
    for k, v in item_list:
        logfile.write('AUC-{},{:.8f}\n'.format(k, v))
    logfile.close()
    print('Wrote results to {}.'.format(logfile_name))

def record_results(logdir, aucs, baseline_results):
    lanczos_results = {'lanczos-min': min(aucs),
                       'lanczos-max': max(aucs),
                       'lanczos-mean': np.mean(aucs),
                       'lanczos-1': aucs[0],
                       'lanczos-middle': aucs[len(aucs) // 2],
                       'lanczos-last': aucs[-1]}
    record_aucs(logdir, 'results.csv',
            list(baseline_results.items()) + list(lanczos_results.items()))


def get_saved_lanczos_tensors(tensordir):
    """Load saved tensors from a previous run of the Lanczos iteration. This
       script saves these tensors as intermediate progress.

    Args:
        tensordir (str): path where these tensors are located.
    Returns:
        saved_tensors (dict): dictionary with saved tensors, with the elements
            of the tridiagonal matrix alpha and beta, as well as the
            associated orthonormal basis Q_lan.
    """
    saved_tensors = {}
    tnames = ['Q_lan', 'Alpha', 'Beta']
    if not os.path.exists(os.path.join(tensordir, 'lanczos_Beta.npy')):
        return None
    for tname in tnames:
        fname = os.path.join(tensordir, 'lanczos_{}.npy'.format(tname))
        saved_tensors[tname] = np.load(fname)
    return saved_tensors

def main(params):
    """Probably being called by run_local_ensembles_main.py.
    Runs the local ensembles (LE) method on a pretrained model, using
    the Lanczos iteration to estimate the top eigenvectors.

    Args:
        params (dict): a dictionary with strings as keys, containing parameters
                       for this program.
            expname (str): name of this experiment
            expdir (str): path where the pretrained model is stored.
            num_examples (int): how many saved examples to run LE on.
            batch_size (int): for dataset iterators.
            num_lanczos_iterations (int): how long to run Lanczos iteration.
            projection_step (int): project gradient onto 1...m-th eigenvector
                for every value of m skipping this many values.
            model_type (str): CNN_classifier or MLP_regressor.
            ckpt_name (str): name of the checkpoint to load.
            two_reorth (bool): whether or not to do second reorthogonalization
                step in Lanczos iteration.
            tf_seed (int): seed for Tensorflow.
            np_seed (int): seed for Numpy.
            run_baselines (bool): whether or not to run the baselines.
    """
    start_time = time.time()
    tf.compat.v1.random.set_random_seed(params['tf_seed'])
    np.random.seed(params['np_seed'])

    expdir = params['expdir']
    expname = params['expname']
    figdir = os.path.join(expdir, expname, 'figs')
    logdir = os.path.join(expdir, expname, 'logs')
    tensordir = os.path.join(expdir, expname, 'tensors')
    for directory in [figdir, logdir, tensordir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Load trained model.
    model = load_model.load_model(expdir, params['model_type'], 
            params['ckpt_name'])
    pred_fn = so.make_pred_fn(model, params['model_type'])
    loss_fn = so.make_loss_fn(model, None)
    grad_fn = so.make_grad_fn(model)
    map_grad_fn = so.make_map_grad_fn(model)

    # Load data.
    print('Loading data.', time.time() - start_time)
    itr_training_data, itr_valid, itr_test, itr_ood = load_saved_data(
            expdir, params['num_examples'], params['batch_size'])

    # Run baselines.
    if params['run_baselines']:
        print('Running baselines.', time.time() - start_time)
        baseline_results = run_baselines(model, itr_valid, itr_test, itr_ood,
                                         params['batch_size'])
        record_aucs(logdir, 'baseline_results.csv', 
                baseline_results.items())
    else:
        print('Skipping baselines.')
        baseline_results = {}
        inputs, _ = itr_training_data.next() # initializing model weights.
        _ = model(inputs)

    # Run Lanczos iteration to tridiagonalize implicit Hessian.
    model_dtype = (np.float64 if params['model_type'] == 'MLP_regressor'
                   else np.float32)
    saved_lanczos_tensors = get_saved_lanczos_tensors(tensordir)
    if saved_lanczos_tensors is None:
        def implicit_hvp(vector):
            vector = tu.reshape_vector_as(model.weights, vector.T)
            hvp = so.hvp(vector, itr_valid, loss_fn, grad_fn, map_grad_fn,
                         n_samples=5)
            hvp_concat = tu.flat_concat(hvp)
            return tf.transpose(hvp_concat).numpy()

        print('Beginning Lanczos iteration.', time.time() - start_time)
        ortho_basis_vectors, beta, alpha = L.lanczos_iteration(
            implicit_hvp, 
            get_model_dim(model), 
            params['num_lanczos_iterations'], 
            eps=EPS,
            dtype=model_dtype, 
            two_reorth=params['two_reorth'])
        ortho_basis = np.concatenate(ortho_basis_vectors[1:-1], axis=1)
        utils.save_tensors([(ORTHO_BASIS_NAME, ortho_basis),
                            (BETA_NAME, np.array(beta).squeeze()),
                            (ALPHA_NAME, np.array(alpha).squeeze())],
                           tensordir)
    else:
        print('loading previously run lanczos vectors')
        ortho_basis = saved_lanczos_tensors['Q_lan']
        alpha = saved_lanczos_tensors['Alpha']
        beta = saved_lanczos_tensors['Beta']
        del saved_lanczos_tensors

    # Find eigenvalues + vectors of tridiagonal + sort by abs(eigenvalue).
    (tridiag_eigenvalues,
     tridiag_eigenvectors) = L.get_eigendecomposition_from_tridiagonal(
         alpha[1:], beta[1:-1])
    (tridiag_eigenvalues, 
     tridiag_eigenvectors) = L.sort_eigendata_by_absolute_value(
         tridiag_eigenvalues, tridiag_eigenvectors)

    # If algorithm terminated early, remove smallest estimate.
    early_terminated = False
    if beta[-1] < EPS:
        tridiag_eigenvalues = tridiag_eigenvalues[:-1]
        tridiag_eigenvectors = tridiag_eigenvectors[:, :-1]
        early_terminated = True

    # Derive the eigenvectors of Hessian from the eigenvector of the
    # tridiagonal matrix.
    matmul_result_path = os.path.join(tensordir, '{}.npy'.format(EVECS_NAME))
    if os.path.exists(matmul_result_path):
        print('Loading {}.'.format(EVECS_NAME), time.time() - start_time)
        del tridiag_eigenvectors
        del ortho_basis
        top_eigenvectors = np.load(matmul_result_path)
    else:
        print('Calculating {}.'.format(EVECS_NAME))
        # Looping can be faster for large models.
        # For smaller models can just do one line instead of this loop: 
        # top_eigenvectors = np.matmul(ortho_basis, tridiag_eigenvectors) 
        ortho_basis = np.split(ortho_basis,
                               np.arange(CHUNK, ortho_basis.shape[0], CHUNK))
        for i in range(len(ortho_basis)):
            print(i)
            if early_terminated: # probably a smaller model, this is safe
                ortho_basis[i] = np.matmul(ortho_basis[i], tridiag_eigenvectors)
            else:
                np.matmul(ortho_basis[i], tridiag_eigenvectors, 
                          out=ortho_basis[i])
        print('Done looping.', time.time() - start_time)
        del tridiag_eigenvectors
        top_eigenvectors = np.concatenate(ortho_basis, axis=0)
        del ortho_basis
        utils.save_tensors([(EVECS_NAME, top_eigenvectors)], tensordir)
    print('Done concatentation.', time.time() - start_time)
    _, top_eigenvectors = L.sort_eigendata_by_absolute_value(
        tridiag_eigenvalues, top_eigenvectors)

    # Use eigen-estimates to perform some OOD-type task by projecting gradients
    # onto estimated eigenvectors, calculating our extrapolation score.
    print('{:d} eigenvectors estimated with Lanczos iteration. '
          'Model has {:d} parameters.'
          .format(top_eigenvectors.shape[1], top_eigenvectors.shape[0]))

    if params['use_prediction_gradient']:
        # this is only implemented for regression models or binary classifiers
        assert (params['model_type'] == 'MLP_regressor' or 
                params['n_labels'] == 2)
        n_labels = 1
        y_values = None
    else:
        if params['model_type'] == 'CNN_classifier':
            n_labels = itr_valid.y.shape[1]
            yvalues = np.eye(n_labels)
        else:
            n_labels = params['n_labels']
            yvalues = np.linspace(np.min(itr_valid.y), np.max(itr_valid.y), 
                    num=n_labels)

    # We'll calculate our score for an increasing number of eigenvectors.
    cutoff_list = np.arange(1, top_eigenvectors.shape[1] + 2, 
                            params['projection_step'])
    extrapolation_scores = {dist: {i: np.zeros((itr.x.shape[0], n_labels))
                for i in cutoff_list}
           for itr, dist in [(itr_test, 'test'), (itr_ood, 'ood')]}

    # Calculate extrapolation score for data from training,
    # in-distribution test data and OOD data.
    for itr, dist in [(itr_test, 'test'), (itr_ood, 'ood')]:
        print('Starting projections for {}.'.format(dist),
              time.time() - start_time)
        for i in range(n_labels):
            if params['use_prediction_gradient']:
                grads = get_pred_grads_in_minibatches(itr.x, pred_fn, 
                        map_grad_fn, params['batch_size'])
            else:
                y = np.tile(yvalues[i], [itr.x.shape[0], 1])
                grads = get_loss_grads_in_minibatches(itr.x, y, loss_fn, 
                        map_grad_fn, params['batch_size'])
            for cutoff in cutoff_list:
                # recalculate full thing each time - no caching
                big_est_evecs = top_eigenvectors[:, :cutoff]
                proj_grads_coeff = np.matmul(grads, big_est_evecs)
                proj_grads = np.matmul(proj_grads_coeff, big_est_evecs.T)
                del proj_grads_coeff
                small_grads = grads - proj_grads
                del proj_grads
                del big_est_evecs
                extrapolation_scores[dist][cutoff][:, i] = np.linalg.norm(
                        small_grads, axis=1)
                del small_grads
            del grads

    # Post-processing - calculate metrics from our extrapolation scores.
    # Loop over results we collected, store results and calculate AUC.
    aucs = calculate_aucs(extrapolation_scores, cutoff_list, tensordir)

    # plot the AUCs across projection iterations
    plot_aucs(figdir, cutoff_list, aucs)

    # Record results to file.
    record_results(logdir, aucs, baseline_results)

