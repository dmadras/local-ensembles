"""Code for 'Visualizing Extrapolation Detection' experiment in the paper 
(Figure 3). Similar to local_ensembles_demo.ipynb, but here you can run
the full ensemble."""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import tensorflow as tf
import pdb
from influence import calculate_influence as ci
from utils import tensor_utils as tu
import local_ensembles_demo_lib as LE
from local_ensembles import lanczos_functions as L
from local_ensembles import evaluation_functions as eval_fns
from utils import utils
import os
mainfigdir = 'iclr_figs/'
figdir = utils.make_subdir(mainfigdir, 'small_NN_tanh_all_test')

# random seeding
seed = 12345
tf.compat.v1.random.set_random_seed(seed)
np.random.seed(seed)

# plotting hyperparameters
legendsize=20
ticksize=22
markersize=30

itr_train, itr_test, itr_ood = LE.setup_data_1d(figdir, legendsize=legendsize, ticksize=ticksize)
itr_test_x = []
for _ in range(4):
    xb, _ = itr_test.__next__()
    itr_test_x.append(xb)
itr_test_x = np.concatenate(itr_test_x, axis=0)
itr_ood_x = []
for _ in range(4):
    xb, _ = itr_ood.__next__()
    itr_ood_x.append(xb)
itr_ood_x = np.concatenate(itr_ood_x, axis=0)

ensemble = []
n_models = 20
n_steps = 400
activ = 'tanh'
hidden_layer_sizes = [3, 3]
for _ in range(n_models):
    model_i = LE.train_NN(itr_train, itr_test, itr_ood, 1, n_steps, activ, hidden_layer_sizes)
    ensemble.append(model_i)
    LE.plot_data_1d_ensemble(ensemble, itr_train, itr_test, itr_ood,
                                  figdir, legendsize=legendsize, ticksize=ticksize)
    model = ensemble[0]
    LE.plot_data_1d(model, itr_train, itr_test, itr_ood, figdir, legendsize=legendsize, ticksize=ticksize)

big_res = {}
for model_ix in range(len(ensemble)):
    print('Running model {:d}'.format(model_ix))
    modelfigdir = utils.make_subdir(figdir, 'model_{:d}'.format(model_ix))
    model = ensemble[model_ix]
    hessian = LE.estimate_Hessian(model, itr_train)
    assert np.linalg.norm(hessian - hessian.T) < 1e-8 # check you estimated a symmetrical Hessian
    A = hessian
    dim = hessian.shape[0]

    # get the true eigendecomposition of the Hessian
    true_evals, true_evecs = np.linalg.eig(A)
    for i in range(dim):
        v = true_evecs[:, i]
        w = np.matmul(A, v)
        assert np.linalg.norm(w - true_evals[i] * v) < 1e-10 # assert it's correct

    # create and test the implicit HVP function
    pred_fn = ci.make_pred_fn(model, 'MLP_regressor')
    loss_fn = ci.make_loss_fn(model, None)
    grad_fn = ci.make_grad_fn(model)
    map_grad_fn = ci.make_map_grad_fn(model)
    explicit_hvp = lambda v: np.matmul(A, v)

    def implicit_hvp(v):
        v = tu.reshape_vector_as(model.weights, v.T)
        hvp = ci.hvp(v, itr_train, loss_fn, grad_fn, map_grad_fn, n_samples=10)
        hvp_concat = tu.flat_concat(hvp)
        return tf.transpose(hvp_concat).numpy()

    # run the Lanczos algorithm
    EPS = 1e-8
    num_iters = dim

    A_evals_all = []
    A_evecs_all = []
    Q, Beta, Alpha = L.lanczos_iteration(implicit_hvp, dim, num_iters, eps=EPS, two_reorth=True, dtype=np.float64)
    Q_lan = np.concatenate(Q[1:-1], axis=1)
    T_evals, T_evecs = L.get_eigendecomposition_from_tridiagonal(Alpha[1:], Beta[1:-1])
    T_evals, T_evecs = L.sort_eigendata_by_absolute_value(T_evals, T_evecs)

    # if algorithm terminated early, remove smallest estimate
    if Beta[-1] < EPS:
        T_evals = T_evals[:-1]
        T_evecs = T_evecs[:,:-1]

    # and derive the eigenvectors of A from the eigenvectors of T
    A_evecs = np.matmul(Q_lan, T_evecs)
    _, A_evecs = L.sort_eigendata_by_absolute_value(T_evals, A_evecs)
    # A and T will have the same eigenvalues
    A_evals = T_evals[:]
    A_evals_all.append(A_evals)
    A_evecs_all.append(A_evecs)

    # PLOTTING ESTIMATED EIGENVALUES
    true_evals, true_evecs = L.sort_eigendata_by_absolute_value(true_evals, true_evecs)

    # plot to see my estimated eigenvalues are correct
    plt.clf()
    plt.plot(sorted(T_evals, reverse=True), label='Lanczos', lw=6, ls='-')
    plt.plot(sorted(true_evals, reverse=True), label='True', lw=3, ls=':')
    plt.legend(prop={'size': legendsize})
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.tight_layout()
    plt.savefig(os.path.join(modelfigdir, 'eigenvalue_accuracy.pdf'))

    # calculate cosine similarity to ground truth eigenvectors
    angles = []
    for i in range(len(T_evals)):
        val = A_evals[i]
        vec_A = A_evecs[:,i]
        vec_true = true_evecs[:, i]
        val_true = true_evals[i]
        print('My {:d}-th eigenvalue: {:.6f} vs. {:.6f}'.format(i, 
                              val, val_true))
        angle = np.matmul(vec_A, vec_true)
        print('Cosine = {:.3f}'.format(angle))
        angles.append(abs(angle))


    numvals = 10
    aucs = []
    for cutoff in range(A_evecs.shape[1]):
        big_est_evecs = A_evecs[:,:cutoff]
        res = {}
        for itr, nm in [(itr_test, 'test'), (itr_ood, 'ood')]:
            print(nm)
            small_grad_norms = []
            x = []
            for _ in range(4):
                xb, _ = itr.__next__()
                x.append(xb)
            x = np.concatenate(x, axis=0)
            # pred grads
            loss_grads = tu.flat_concat(ci.get_pred_grads(x, pred_fn, map_grad_fn))
            proj_grads_coeff = np.matmul(loss_grads, big_est_evecs)
            proj_grads = np.matmul(proj_grads_coeff, big_est_evecs.T)
            small_grads = loss_grads - proj_grads
            small_grad_norms = np.linalg.norm(small_grads, axis=1, keepdims=True)
            res[nm] = (small_grad_norms, x)

        plt.clf()
        plt.scatter(res['test'][1], res['test'][0], label='InD', marker='o', s=markersize)
        plt.scatter(res['ood'][1], res['ood'][0], label='OOD', marker='*', s=markersize)
        plt.legend(prop={'size': legendsize})
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        plt.tight_layout()
        plt.savefig(os.path.join(modelfigdir, 'auc_{:d}.pdf'.format(cutoff)))
                            
        auc = eval_fns.get_auc(res['test'][0], res['ood'][0])
        aucs.append(auc)
        print('{:d} E-vecs: AUC = {:.2f}'.format(cutoff, auc))

    plt.clf()
    fig, ax1 = plt.subplots()
    t = np.arange(len(angles))
    color = 'tab:red'
    # ax1.set_xlabel('Number of projections',  fontsize=15)
    ax1.set_ylabel('AUC', color=color, fontsize=20)
    ax1.plot(t, aucs, color=color, lw=4)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.yticks(fontsize=ticksize)
    plt.xticks(fontsize=ticksize - 8)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Cosine Similarity', color=color, fontsize=20)  # we already handled the x-label with ax1
    ax2.plot(t, angles, color=color, ls=':', lw=4)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.tight_layout()
    plt.yticks(fontsize=ticksize)
    plt.savefig(os.path.join(modelfigdir, 'auc_vs_angles.pdf'), bbox_inches="tight")

    cutoff = np.argmax(aucs)
    print('Best AUC was with {:d} eigenvectors'.format(cutoff))
    big_est_evecs = A_evecs[:,:cutoff]
    res = {}
    for x, nm in [(itr_test_x, 'test'), (itr_ood_x, 'ood')]:
        print(nm)
        small_grad_norms = []
        # pred grads
        loss_grads = tu.flat_concat(ci.get_pred_grads(x, pred_fn, map_grad_fn))
        proj_grads_coeff = np.matmul(loss_grads, big_est_evecs)
        proj_grads = np.matmul(proj_grads_coeff, big_est_evecs.T)
        small_grads = loss_grads - proj_grads
        small_grad_norms = np.linalg.norm(small_grads, axis=1, keepdims=True)
        res[nm] = (small_grad_norms, x)
    big_res[model_ix] = res


res_y_test = np.mean(np.concatenate([big_res[i]['test'][0] for i in big_res], axis=1), axis=1)
res_y_ood = np.mean(np.concatenate([big_res[i]['ood'][0] for i in big_res], axis=1), axis=1)
res_y = np.concatenate([res_y_test, res_y_ood], axis=0)
res_x = np.concatenate([itr_test_x, itr_ood_x], axis=0)

res_x_list = [res_x[i] for i in range(len(res_x))]
res_y_list = [res_y[i] for i in range(len(res_y))]
res_sorted = sorted(zip(res_x_list, res_y_list), key=lambda p: p[0])
res_x = np.array([p[0] for p in res_sorted])
res_y = np.array([p[1] for p in res_sorted])

plt.clf()
plt.scatter(itr_test_x, res_y_test, label='InD', marker='o', s=markersize)
plt.scatter(itr_ood_x, res_y_ood, label='OOD', marker='*', s=markersize)
plt.legend(prop={'size': legendsize})
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.tight_layout()
plt.savefig(os.path.join(figdir, 'scores_ind_vs_ood.pdf'.format(cutoff)))

outputs = []
xrange = np.expand_dims(np.arange(-3, 4, 0.1), 1)
for mdl in ensemble:
    xrange_output, _ = mdl(xrange)
    outputs.append(xrange_output)
outputs = tf.concat(outputs, axis=1)
output_stdev = np.std(outputs, axis=1)

plt.clf()
fig, ax1 = plt.subplots()
color = 'tab:red'
# ax1.set_xlabel('Number of projections',  fontsize=15)
ax1.set_ylabel('Score', color=color, fontsize=20)
ax1.plot(res_x, res_y, color=color, lw=4)
ax1.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize=ticksize)
plt.xticks(fontsize=ticksize - 8)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Standard Deviation', color=color, fontsize=20)  # we already handled the x-label with ax1
ax2.plot(xrange, output_stdev, color=color, ls=':', lw=4)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.yticks(fontsize=ticksize)
plt.savefig(os.path.join(figdir, 'score_vs_stdev.pdf'), bbox_inches="tight")


def inbin(x, lo, hi):
    return np.logical_and(x.flatten() >= lo, x.flatten() < hi)

bins_lo = np.linspace(min(xrange), max(xrange), 20)
bins_hi = bins_lo + 1

stdevs_in_bin = [np.mean(output_stdev[inbin(xrange, bins_lo[i], bins_hi[i])]) 
                    for i in range(len(bins_lo))]
scores_in_bin = [np.mean(res_y[inbin(res_x, bins_lo[i], bins_hi[i])])
                    for i in range(len(bins_lo))]
plt.clf()
plt.scatter(stdevs_in_bin, scores_in_bin, s=markersize)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.tight_layout()
plt.savefig(os.path.join(figdir, 'score_vs_stdev_bins.pdf'))

