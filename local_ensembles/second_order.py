"""Functions for doing second-order manipulation of pretrained models.

Some code adapted from https://github.com/kohpangwei/influence-release/blob/
master/influence/genericNeuralNet.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def get_loss_grads(x, y, loss_fn, map_grad_fn):
  with tf.GradientTape(persistent=True) as tape:
    loss = loss_fn(x, y)
    grads = map_grad_fn(loss, tape)
  return grads

def get_pred_grads(x, pred_fn, map_grad_fn):
  with tf.GradientTape(persistent=True) as tape:
    preds = pred_fn(x)
    grads = map_grad_fn(preds, tape)
  return grads

def hvp(v, iterator, loss_fn, grad_fn, map_grad_fn, n_samples=1):
  """Multiply the Hessian of clf at inputs (x, y) by vector v.

  Args:
    v (tensor): the vector in the HVP.
    iterator (Iterator): iterator for samples for HVP estimation.
    loss_fn (function): a function which returns a gradient of losses.
    grad_fn (function): a function which takes the gradient of a scalar loss.
    map_grad_fn (function): a function which takes the gradient of each element
                            of a vector of losses.
    n_samples (int, optional): number of minibatches to sample
                               when estimating Hessian
  Returns:
    hessian_vector_val (tensor): the HVP of clf's Hessian with v.
  """

  # tf.GradientTape tracks the operations you take while inside it, in order to
  # later auto-differentiate through those operations to get gradients.
  with tf.GradientTape(persistent=True) as tape2:

    # We need two gradient tapes to calculate second derivatives
    with tf.GradientTape() as tape:
      loss = 0.
      for _ in range(n_samples):
        x_sample, y_sample = iterator.next()
        loss += tf.reduce_mean(loss_fn(x_sample, y_sample)) 
      loss /= n_samples

    # Outside the tape, we can get the aggregated loss gradient across the
    # batch. This is the standard usage of GradientTape.
    grads = grad_fn(loss, tape)

    # For each weight matrix, we now get the product of the vector v with
    # the gradient, and the sum over the weights to get a total gradient
    # per element in x.
    vlist = []
    for g, u in zip(grads, v):
      g = tf.expand_dims(g, 0)
      prod = tf.multiply(g, u)
      vec = tf.reduce_sum(prod, axis=range(1, prod.shape.rank))
      vlist.append(vec)
    vgrads = tf.add_n(vlist)

    # We now take the gradient of the gradient-vector product. This gives us
    # the Hessian-vector product. Note that we take this gradient inside
    # the tape - this allows us to get the HVP value for each element of x.
    hessian_vector_val = map_grad_fn(vgrads, tape2)
  return hessian_vector_val


def make_pred_fn(clf, model_type):
  """Return a function which returns a vector of per-example predictions.

  Args:
    clf (Classifier): the classifier whose loss we are interested in. Output
                    is a two-column Bernoulli distribution ('CNN_classifier')
                    or a scalar ('MLP_regressor').
  Returns:
    f (function): a function which runs clf on input x and output y and returns
                  a vector of predictions, one for each element in x.
  """
  def f(x):
      preds, _ = clf(x)
      if model_type == 'CNN_classifier':
          return tf.reshape(preds[:,1], [-1, 1])
      else:
          return preds
  return f

def make_loss_fn(clf, lam):
  """Return a function which returns a vector of per-examples losses.

  Args:
    clf (Classifier): the classifier whose loss we are interested in.
    lam (float): optional L2 regularization parameter.
  Returns:
    f (function): a function which runs clf on input x and output y and returns
                  a vector of losses, one for each element in x.
  """
  if lam is None:
    def f(x, y):
      train_loss, _ = clf.get_loss(x, y)
      return train_loss
  else:
    def f(x, y):
      train_loss = clf.get_loss_dampened(x, y, lam=lam)
      return train_loss
  return f


def make_grad_fn(clf):
  """Return a function which takes the gradient of a loss.

  Args:
    clf (Classifier): the classifier whose gradient we are interested in.
  Returns:
    f (function): a function which takes a scalar loss and GradientTape and
                  returns the gradient of loss w.r.t clf.weights.
  """
  def f(loss, tape):
    return tape.gradient(loss, clf.weights, 
			unconnected_gradients=tf.UnconnectedGradients.ZERO)
  return f


def make_map_grad_fn(clf):
  """Return a function which takes the gradient of each element of loss vector.

  Args:
    clf (Classifier): the classifier whose gradient we are interested in.
  Returns:
    f (function): a function which takes a vector v and a GradientTape and
                  takes the gradient on the tape for each element of v.
  """
  def f(v, tape):
    return tf.map_fn(lambda l:
            tape.gradient(l, clf.weights, unconnected_gradients=tf.UnconnectedGradients.ZERO),
            v,
            dtype=tf.nest.map_structure(lambda x: x.dtype, clf.weights))
  return f

