"""Dataset-related utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

_THRESHOLD = 0.7
NUM_CLASSES = {"mnist": 10, "fashion_mnist": 10, "celeb_a": 2}
PROCESS_FUNCTIONS = {"mnist": 'greyscale', 
                     "fashion_mnist": 'greyscale', 
                     "celeb_a": 'colour'}
HASH_ARGS = {"train": {"split_name": "train", "hash_args": [5, (0, 1, 2, 3)]},
             "valid": {"split_name": "train", "hash_args": [5, (4,)]},
             "test": {"split_name": "test", "hash_args": [1, (0,)]}}

def filter_label_function(classes):
  """Return a function which returns true if an examples label is in classes.

  Args:
    classes (list): which classes to keep.
  Returns:
    f (function): a function for filtering to keep only examples in classes.
  """
  classes_array = np.array(classes).astype(np.int64)
  def f(x):
    return tf.math.reduce_any(tf.equal(classes_array, x["label"]))
  return f

def binarize_image_and_make_label_onehot(threshold, n_classes):
  """Returns a function for binarizing a black and white image from raw data.

  Args:
    threshold (float): cutoff for binarizing pixels, between 0 and 1.
  Returns:
    f (function): image processing function.
  """
  def f(x):
    """Returns a binarized version of a greyscale image and its label.

    Args:
      x (float): where to cut off between white and black.
    Returns:
      image, label (tuple): binarized image and its associated one-hot label.
    """
    img = tf.round(tf.cast(x["image"], "float") / 255. / 2. / threshold)
    label = x["label"]
    return (img, make_onehot(label, n_classes))
  return f

def process_colour_image_and_make_label_onehot(n_classes):
  """Returns a function for processing a colour image.  """
  def f(x):
    img = tf.cast(x["image"], "float") / 255.
    label = x["label"]
    return (img, make_onehot(label, n_classes))
  return f

def get_image_processing_function(dataset_name):
  if PROCESS_FUNCTIONS[dataset_name] == 'greyscale':
    process_fn = binarize_image_and_make_label_onehot(
                          _THRESHOLD, NUM_CLASSES[dataset_name])
  elif PROCESS_FUNCTIONS[dataset_name] == 'colour':
    process_fn = process_colour_image_and_make_label_onehot(
                        NUM_CLASSES[dataset_name])
  return process_fn

def get_hash(x, buckets=5):
  s = tf.strings.as_string(x)
  s_joined = tf.strings.reduce_join(s)
  return tf.strings.to_hash_bucket(s_joined, num_buckets=buckets)

def get_hash_filter_function(buckets, keep_vals):
  """Return a function which filters an (x, y) pair by its hash value.
  This is used to randomly split out a validation set from the training set
  defined in tfds. The validation set will be of size
  (len(keep_vals) / buckets) * (original training set size).
  Args:
    buckets (int): how many buckets to hash into.
    keep_vals (list of ints): which hash values we want to keep.
  Returns:
    f (function): a function which filters an (x, y) pair by its hash value.
  """
  def f(x):
    xinp = tf.concat([tf.reshape(tf.cast(x["image"], tf.float32), [-1, 1]),
                      tf.reshape(tf.cast(x["label"], tf.float32), [-1, 1])],
                     axis=0)
    hash_value = get_hash(xinp, buckets=buckets)
    return tf.reduce_any(tf.equal(hash_value, keep_vals))
  return f

def make_onehot(label, depth):
  """Make integer tensor label into a one-hot tensor.

  Args:
    label (tensor): a single-column tensor of integers.
    depth (int): the number of categories for the one-hot tensor.
  Returns:
    onehot (tensor): a one-hot float tensor of the same length as label.
  """
  return tf.squeeze(tf.one_hot(label, depth=depth))

def get_iterator_by_class(dataset_name, phase, classes, process_function, 
        batch_size=32, tfds_path=os.environ['TFDS_PATH'], shuffle_buffer_size=1024):
  """Get an iterator for phase = {train, valid, test}."""
  dataset_itr = (
      tfds.load(dataset_name, split=HASH_ARGS[phase]["split_name"],
                as_dataset_kwargs={"shuffle_files": False}, 
                data_dir=tfds_path)
      .filter(get_hash_filter_function(*HASH_ARGS[phase]["hash_args"]))
      .filter(filter_label_function(classes))
      .map(process_function)
      .shuffle(shuffle_buffer_size).repeat().batch(batch_size).prefetch(4))
  return tf.compat.v1.data.make_one_shot_iterator(dataset_itr)

def load_dataset_ood(dataset_name, ind_classes, ood_classes, 
                        process_function, batch_size=32):
  """Load dataset with OOD splits.

  Args:
    dataset_name (str): the name of this dataset.
    ind_classes (list): ints of which classes we want to train on.
    ood_classes (list): ints of which classes we want evaluate OOD on.
    process_function (dict): parameters for loading iterators.
  Returns:
    four iterators, for training, validation, testing, and OOD evaluation.
  """
  return (get_iterator_by_class(dataset_name, "train", ind_classes,
                                process_function, batch_size=batch_size),
          get_iterator_by_class(dataset_name, "valid", ind_classes,
                                process_function, batch_size=batch_size),
          get_iterator_by_class(dataset_name, "test", ind_classes,
                                process_function, batch_size=batch_size),
          get_iterator_by_class(dataset_name, "test", ood_classes,
                                process_function, batch_size=batch_size))

def load_dataset_ood_supervised_onehot(ind_classes, ood_classes,
                                        dataset_name="mnist", batch_size=32):
  """Load dataset with OOD splits.

  Args:
    ind_classes (list): ints of which classes we want to train on.
    ood_classes (list): ints of which classes we want evaluate OOD on.
    dataset_name (str): the name of this dataset.
    label_noise (float): percentage of data we want to add label noise for.
  Returns:
    Four iterators, for training, validation, testing, and OOD evaluation.
  """
  process_fn = get_image_processing_function(dataset_name)
  return load_dataset_ood(dataset_name, ind_classes, ood_classes, process_fn, 
          batch_size=batch_size)

'''Data-related functions for testing.'''

def get_supervised_batch_noise(x_shape, y_shape):
  # For one-hot encoding, we want to test float tensors.
  # For integer labeling, we want to test integer tensors.
  y_type = tf.dtypes.int32 if y_shape[1] == 1 else tf.dtypes.float32
  return (tf.random.uniform(x_shape), tf.ones(y_shape, dtype=y_type))


def get_supervised_batch_noise_iterator(x_shape, y_shape=None):
  """Return an iterator which returns noise, for testing purposes.

  Args:
    x_shape (tuple): shape of x data to return.
    y_shape (tuple): shape of y data to return.
  Returns:
    itr (TestIterator): an iterator which returns batches in the given shapes.
  """

  class TestIterator(object):
    """An iterator which returns noise, for testing purposes."""

    def __init__(self, x_shape, y_shape):
      self.x_shape = x_shape
      self.y_shape = y_shape

    def __iter__(self):
      return self

    def next(self):
      if self.y_shape is not None:
        return get_supervised_batch_noise(x_shape, y_shape)
      else:
        x, _ = get_supervised_batch_noise(x_shape, (1, 1))
        return x

  return TestIterator(x_shape, y_shape)

