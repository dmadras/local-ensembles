"""Train a classification model for an OOD-detection-style task.

We train a classifier on some dataset, holding out some subset of classes.
We also save a subset of the train, valid, test, and OOD splits to disk so we
can access them later in an easier way.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
from model import classifier
from model import train_model
from utils import dataset_utils
from utils import utils

flags.DEFINE_string('expname', 'temp', 'name of this experiment directory')
flags.DEFINE_integer('max_steps_train', 1000, 'number of steps of optimization')
flags.DEFINE_integer('max_steps_test', 10, 'number of steps of testing')
flags.DEFINE_integer('run_avg_len', 50,
                     'number of steps of average losses over')
flags.DEFINE_integer('write_freq', 50, 'number of steps between printing')
flags.DEFINE_float('lr', 0.001, 'Adam learning rate')
flags.DEFINE_string('conv_dims', '80,40,20',
                    'comma-separated list of integers for conv layer sizes')
flags.DEFINE_string('conv_sizes', '5,5,5',
                    'comma-separated list of integers for conv filter sizes')
flags.DEFINE_string('dense_sizes', '100',
                    'comma-separated list of integers for dense hidden sizes')
flags.DEFINE_integer('patience', 50, 'steps of patience for early stopping')
flags.DEFINE_integer('tf_seed', 0, 'random seed for Tensorflow')
flags.DEFINE_integer('np_seed', 0, 'random seed for Numpy')
flags.DEFINE_integer('n_classes', 10, 'number of classes in prediction')
flags.DEFINE_string('training_results_dir',
                    '/tmp',
                    'Output directory for experimental results.')
flags.DEFINE_string('ood_classes', '5', 'a comma-separated list of'
                    'which labels to consider OoD. Empty string means'
                    'no classes are OoD.')
flags.DEFINE_string('dataset_name', 'mnist', 'what dataset to use')
flags.DEFINE_integer('n_test_save', 1000, 'how many test examples to save')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  params = FLAGS.flag_values_dict()
  tf.compat.v1.random.set_random_seed(params['tf_seed'])
  np.random.seed(params['np_seed'])

  params['results_dir'] = utils.make_subdir(
      params['training_results_dir'], params['expname'])
  params['figdir'] = utils.make_subdir(params['results_dir'], 'figs')
  params['ckptdir'] = utils.make_subdir(params['results_dir'], 'ckpts')
  params['logdir'] = utils.make_subdir(params['results_dir'], 'logs')
  params['tensordir'] = utils.make_subdir(params['results_dir'], 'tensors')

  # Load the classification model.
  conv_dims = [int(x) for x in (params['conv_dims'].split(',')
                                if params['conv_dims'] else [])]
  conv_sizes = [int(x) for x in (params['conv_sizes'].split(',')
                                 if params['conv_sizes'] else [])]
  dense_sizes = [int(x) for x in (params['dense_sizes'].split(',')
                                  if params['dense_sizes'] else [])]
  params['n_layers'] = len(conv_dims)
  clf = classifier.CNN(conv_dims, conv_sizes, dense_sizes,
                       params['n_classes'], onehot=True)

  # Checkpoint the initialized model, in case we want to re-run it from there.
  utils.checkpoint_model(clf, params['ckptdir'], 'initmodel')

  # Load the "in-distribution" and "out-of-distribution" classes as
  # separate splits.
  all_classes = range(params['n_classes'])
  if params['ood_classes'] != '':
      ood_classes = [int(x) for x in params['ood_classes'].split(',')]
      # We assume we train on all non-OOD classes.
      ind_classes = [x for x in all_classes if x not in ood_classes]
  else:
      ood_classes = [0] # filler
      ind_classes = [x for x in all_classes]
  (itr_train, itr_valid, itr_test, itr_test_ood
  ) = dataset_utils.load_dataset_ood_supervised_onehot(
      ind_classes, ood_classes, 
      dataset_name=params['dataset_name'])


  # Save to disk samples of size params['n_test_save'] from the train, valid, test and OOD sets.
  train_data = utils.aggregate_batches(itr_train, params['n_test_save'],
                                       ['train_x_infl', 'train_y_infl'])

  validation_data = utils.aggregate_batches(itr_valid, params['n_test_save'],
                                            ['valid_x_infl', 'valid_y_infl'])

  test_data = utils.aggregate_batches(itr_test, params['n_test_save'],
                                      ['test_x_infl', 'test_y_infl'])

  ood_data = utils.aggregate_batches(itr_test_ood, params['n_test_save'],
                                     ['ood_x_infl', 'ood_y_infl'])
  utils.save_tensors(list(train_data.items()) + list(validation_data.items()) +
                     list(test_data.items()) + list(ood_data.items()),
                     params['tensordir'])

  # Train and test the model in-distribution, and save test outputs.
  train_model.train_classifier(clf, itr_train, itr_valid, params)
  train_model.test_classifier(clf, itr_test, params, 'test')

  # Save model outputs on the training set.
  params['tensordir'] = utils.make_subdir(
      params['results_dir'], 'train_tensors')
  train_model.test_classifier(clf, itr_train, params, 'train')

  # Save model outputs on the OOD set.
  params['tensordir'] = utils.make_subdir(
      params['results_dir'], 'ood_tensors')
  train_model.test_classifier(clf, itr_test_ood, params, 'ood')

  params['tensordir'] = utils.make_subdir(
      params['results_dir'], 'tensors')

if __name__ == '__main__':
  app.run(main)
