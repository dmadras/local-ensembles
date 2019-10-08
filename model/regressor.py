"""A regression model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import tensorflow as tf
from utils import dataset_utils

FLAGS = flags.FLAGS
EPS = 1e-12


class Regressor(tf.keras.Model):
  """A classifier."""

  def __init__(self):
    """Initializes an n-way classifier.

    Args:
      n_classes (int): number of classes in the problem.
      onehot (bool): whether the labels take one-hot or integer format.
    """
    super(Regressor, self).__init__()

  def build_layers(self):
    """Build the internal layers of the network."""
    raise NotImplementedError

  def call(self, inputs):
    """Feed inputs forward through the network.

    Args:
      inputs (tensor): input batch we wish to encode.
    Returns:
      logits (tensor): the unnormalized output of the model on this batch.
    """
    raise NotImplementedError

  def get_loss(self, batch_x, batch_y, return_preds=False):
    """Run classifier on input batch and return loss, error, and predictions.

    Args:
      batch_x (tensor): input batch we wish to run on.
      batch_y (tensor): onehot input labels we wish to predict.
      return_preds (bool): whether or not to return prediction tensor.
    Returns:
      loss (tensor): cross-entropy loss for each element in batch.
      err (tensor): classification error for each element in batch.
      preds (tensor): optional, model softmax outputs.
      reprs (tensor): optional, internal model representations.
    """
    preds, reprs = self.call(batch_x)
    err = preds - batch_y
    loss = 0.5 * tf.square(err)
    if return_preds:
      return loss, err, preds, reprs
    else:
      return loss, err

  def get_loss_dampened(self, batch_x, batch_y, lam=0.0):
    loss, _ = self.get_loss(batch_x, batch_y, return_preds=False)
    reg_loss = 0.5 * sum([tf.reduce_sum(tf.square(w)) for w in self.weights])
    loss = loss + lam * reg_loss
    return loss

class MLP(Regressor):
  """A fully-connected classifier."""

  def __init__(self, dense_sizes, activation, **kwargs):
    super(MLP, self).__init__(**kwargs)
    self.dense_sizes = dense_sizes
    self.activation = activation
    self.dense_layers = []
    self.build_layers()

  def build_layers(self):
    activation_fn = {'relu': tf.nn.relu,
                      'softplus': tf.nn.softplus,
                      'tanh': tf.nn.tanh,
                     'sigmoid': tf.nn.sigmoid}[self.activation] 
    for dim in self.dense_sizes:
      layer = tf.keras.layers.Dense(dim, activation=activation_fn)
      self.dense_layers.append(layer)
    layer = tf.keras.layers.Dense(1)
    self.dense_layers.append(layer)

  def call(self, inputs):
    reprs = []
    logits = tf.keras.layers.Flatten()(inputs)
    for l in self.dense_layers:
      logits = l(logits)
      reprs.append(logits)
    return logits, reprs

