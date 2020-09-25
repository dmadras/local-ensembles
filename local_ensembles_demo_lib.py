"""Functions for local_ensembles_demo.ipynb and local_ensembles_demo.py.
"""

import os 
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from utils import tensor_utils as tu
from utils.dataset_iterator import DatasetIterator
from local_ensembles import second_order as so

# build small neural network
class MLPRegressor(tf.keras.Model):
    def __init__ (self, layer_dims, output_dims, activ='relu'):
        super(MLPRegressor, self).__init__()
        self.layer_dims = layer_dims
        self.dense_layers = []
        self.activ = {'relu': tf.nn.relu,
                      'softplus': tf.nn.softplus,
                      'tanh': tf.nn.tanh}[activ]
        self.output_dims = output_dims
        self.build_layers()
    
    def build_layers(self):
        for dim in self.layer_dims:
            self.dense_layers.append(
                tf.keras.layers.Dense(dim, activation=self.activ))
        self.dense_layers.append(tf.keras.layers.Dense(self.output_dims))
    
    def call(self, x):
        output = x
        for l in self.dense_layers:
            output = l(output)
        return output, None # compatible with other network models

    def get_loss(self, batch_x, batch_y):
        ypred, _ = self.call(batch_x)
        err = batch_y - ypred
        return tf.square(err), None

# SETUP DATA
def setup_data_1d(figdir, legendsize=20, ticksize=22):
    n_tr = 200
    n_te = 100
    n_oo = 200
    n_batch = 32

    # generate some 1-D data
    train_x = np.concatenate([np.random.uniform(-1, 0, size=(n_tr // 2, 1)), 
                              np.random.uniform(1, 2, size=(n_tr // 2, 1))], axis=0)
    train_y = np.sin(train_x * 4) + np.random.normal(scale=0.5, size=train_x.shape)
    itr_train = DatasetIterator(train_x, train_y, n_batch)

    test_x = np.concatenate([np.random.uniform(-1, 0, size=(n_te // 2, 1)),
                             np.random.uniform(1, 2, size=(n_te // 2, 1))], axis=0)
    test_y = np.sin(test_x * 4) + np.random.normal(scale=0.5, size=test_x.shape)
    itr_test = DatasetIterator(test_x, test_y, n_batch)

    ood_x = np.concatenate([np.random.uniform(-3, -1, size=(n_oo // 4, 1)), 
                            np.random.uniform(0, 1, size=(n_oo // 2, 1)),
                           np.random.uniform(2, 4, size=(n_oo // 4, 1))], axis=0)
    ood_y = np.sin(ood_x * 4) + np.random.normal(scale=0.5, size=ood_x.shape)
    itr_ood = DatasetIterator(ood_x, ood_y, n_batch)

    alpha=0.5
    plt.scatter(train_x, train_y, label='Train', alpha=alpha)
    plt.scatter(test_x, test_y, label='Test', alpha=alpha)
    plt.scatter(ood_x, ood_y, label='OOD', alpha=alpha)
    plt.legend(prop={'size': legendsize})
    plt.tight_layout()
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.savefig(os.path.join(figdir, 'data.pdf'))
    plt.show()

    return itr_train, itr_test, itr_ood
    
def plot_data_1d(model, itr_train, itr_test, itr_ood, figdir, 
			legendsize=20, ticksize=22):
    alpha=0.5
    plt.clf()
    plt.scatter(itr_train.x, itr_train.y, label='Train', alpha=alpha)
    plt.scatter(itr_test.x, itr_test.y, label='Test', alpha=alpha)
    plt.scatter(itr_ood.x, itr_ood.y, label='OOD', alpha=alpha)
    plt.legend(prop={'size': legendsize})

    xrange = np.expand_dims(np.arange(-4, 6, 0.1), 1)
    xrange_output, _ = model(xrange)
    plt.plot(xrange, xrange_output)
    plt.tight_layout()
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.savefig(os.path.join(figdir, 'data_and_model.pdf'))

    plt.show()
    
def plot_data_1d_ensemble(models, itr_train, itr_test, itr_ood, figdir, 
				legendsize=20, ticksize=22):
    alpha=0.5
    plt.clf()
    plt.scatter(itr_train.x, itr_train.y, label='Train', alpha=alpha)
    plt.scatter(itr_test.x, itr_test.y, label='Test', alpha=alpha)
    plt.scatter(itr_ood.x, itr_ood.y, label='OOD', alpha=alpha)
    plt.legend(prop={'size': legendsize})

    xrange = np.expand_dims(np.arange(-4, 6, 0.1), 1)
    for model in models:
        xrange_output, _ = model(xrange)
        plt.plot(xrange, xrange_output)
    plt.tight_layout()
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.savefig(os.path.join(figdir, 'data_and_model_ensemble.pdf'))

    plt.show()

# train the NN
def train_NN(itr_train, itr_test, itr_ood, output_dims, n_steps, activ, hidden_layer_sizes):
    model = MLPRegressor(hidden_layer_sizes, output_dims, activ)
    opt = tf.compat.v1.train.AdamOptimizer(0.01)

    for i in range(n_steps):
        batch_x, batch_y = itr_train.__next__()
        with tf.GradientTape() as tape:
            train_loss, _ = model.get_loss(batch_x, batch_y)
            mean_train_loss = tf.reduce_mean(train_loss)

        batch_x, batch_y = itr_test.__next__()
        test_loss, _ = model.get_loss(batch_x, batch_y)
        batch_x, batch_y = itr_ood.__next__()
        ood_loss, _ = model.get_loss(batch_x, batch_y)

        grads = tape.gradient(mean_train_loss, model.weights)
        opt.apply_gradients(zip(grads, model.weights))

        if i % 20 == 0:
            print('Step {:d}: Test loss = {:.2f}, Train loss = {:.2f}, OOD loss = {:.2f}'
                  .format(i, mean_train_loss, tf.reduce_mean(test_loss), tf.reduce_mean(ood_loss)))

    return model

def estimate_Hessian(model, itr, num_hessian_est=10):
    # get the ground truth Hessian
    loss_fn = so.make_loss_fn(model, None)
    grad_fn = so.make_grad_fn(model)
    map_grad_fn = so.make_map_grad_fn(model)
    with tf.GradientTape(persistent=True) as tape:
        loss_grads_total = 0
        for i in range(num_hessian_est):
            print('Estimating Hessian: minibatch {:d}'.format(i))
            x, y = itr.__next__()
            loss_grads = so.get_loss_grads(x, y, loss_fn, map_grad_fn)
            loss_grads_total += tf.reduce_mean(tu.flat_concat(loss_grads), axis=0)
        loss_grads_total /= num_hessian_est
        print('Taking second derivative - this may be slow...')
        hessian = tu.flat_concat(map_grad_fn(loss_grads_total, tape))

    hessian = hessian.numpy()
    return hessian
