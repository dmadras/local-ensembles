"""Library functions for manipulating tensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


EPS = 1e-10


def reshape_vector_as(weights, vec):
  """Reshape vector vec to look like model parameters weights (batched).

  Let vec be (n x d), and weights be a list of shapes (s0, s1 ... s_p). Then we
  want reshape vec to be a list of tensors shaped (n x s0, n x s1 ... n x s_p),
  so that the first index of every element stays constant. This operation
  is the inverse of flat_concat.

  Args:
    weights (list of tensors): a list of tensors with same number of total
      elements as the second dimension of vec.
    vec (tensor): a matrix (n x d) -- weights has d total elements.
  Returns:
    vec_as_weights (list of tensors): a reshaped version of vec as weights.
  """
  n_ex = vec.shape[0]
  vec_as_weights = []
  curr = 0
  for w in weights:
    num_weights = tf.size(w)
    new_shape = tf.concat([tf.constant([n_ex]), w.shape], 0)
    new_weight = tf.reshape(vec[:, curr:curr + num_weights], new_shape)
    vec_as_weights.append(new_weight)
    curr += num_weights
  return vec_as_weights


def flat_concat(params):
  """Concatenates and flattens a list of tensors.

  This operation is the inverse of reshape_vector_as.

  Args:
    params (list): list of tensors to concatenate and flatten.
  Returns:
    flat_concat_params (tensor): a concatenated and flattened version of params.
  """
  return tf.concat(tf.nest.map_structure(
      lambda x: tf.reshape(x, [x.shape[0], -1]), params), 1)


def cosine_similarity(x, y):
  return tf.reduce_sum(tf.multiply(x, y)) / (tf.linalg.norm(x)
                                             * tf.linalg.norm(y))


def normalize_weight_shaped_vector(weight_shaped_vec):

  flat_vec = flat_concat(weight_shaped_vec)
  norm_vec = tf.maximum(
      tf.math.reduce_euclidean_norm(flat_vec, axis=1, keepdims=True), EPS)
  normalized_vec = flat_vec / norm_vec
  return reshape_vector_as([el[0] for el in weight_shaped_vec], normalized_vec)

def get_data_in_minibatches(itr, mb_size, num_examples):
    x = []
    y = []
    curr = 0
    while curr < num_examples:
        batch_x, batch_y = itr.next() 
        batch_x = batch_x[:min(num_examples - curr, mb_size)] 
        batch_y = batch_y[:min(num_examples - curr, mb_size)] 
        x.append(batch_x)
        y.append(batch_y)
        curr += mb_size
    x = tf.concat(x, axis=0).numpy()
    y = tf.concat(y, axis=0).numpy()
    return x, y
