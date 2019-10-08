"""Train a classifier. Some code due to Jamie Smith."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from absl import logging
import numpy as np
import tensorflow as tf
from utils import utils
from utils.running_average_loss import RunningAverageLoss as RALoss


def train_classifier(clf, itr_train, itr_valid, params):
  """Train a classifier.

  Args:
    clf (classifier): a classifier we wish to train.
    itr_train (Iterator): an iterator over training data.
    itr_valid (Iterator): an iterator over validation data.
    params (dict): flags for training.

  """
  # Dump the parameters we used to a JSON file.
  params_file = os.path.join(params['results_dir'], 'params.json')
  utils.write_json(params_file, params)

  run_avg_len = params['run_avg_len']
  max_steps = params['max_steps_train']
  write_freq = params['write_freq']

  # RALoss is an object which tracks the running average of a loss.
  ra_loss = RALoss('loss', run_avg_len)
  ra_error = RALoss('error', run_avg_len)
  ra_trainloss = RALoss('train-loss', run_avg_len)
  ra_trainerr = RALoss('train-err', run_avg_len)

  min_val_loss = sys.maxsize
  min_val_step = 0
  opt = tf.compat.v1.train.AdamOptimizer(learning_rate=params['lr'])
  finished_training = False
  start_printing = 0
  for i in range(max_steps):
    batch_x, batch_y = itr_train.next()
    with tf.GradientTape() as tape:
      train_loss, train_err = clf.get_loss(batch_x, batch_y)
      mean_train_loss = tf.reduce_mean(train_loss)

    val_batch_x, val_batch_y = itr_valid.next()
    valid_loss, valid_err = clf.get_loss(val_batch_x, val_batch_y)
    loss_list = [ra_loss, ra_error, ra_trainloss, ra_trainerr]
    losses = zip(loss_list,
                 [tf.reduce_mean(l) for l in
                  (valid_loss, valid_err, train_loss, train_err)])
    utils.update_losses(losses)

    grads = tape.gradient(mean_train_loss, clf.weights)
    opt.apply_gradients(zip(grads, clf.weights))

    utils.print_losses(loss_list, i)
    curr_ra_loss = ra_loss.get_value()
    if (curr_ra_loss < min_val_loss and \
            i - min_val_step > params['patience'] / 10) \
            or i == max_steps - 1:
      # Early stopping: stop training when validation loss stops decreasing.
      # The second condition ensures we don't checkpoint every step early on.
      min_val_loss = curr_ra_loss
      min_val_step = i
      save_path, ckpt = utils.checkpoint_model(clf, params['ckptdir'])
      logging.info('Step {:d}: Checkpointed to {}'.format(i, save_path))

    if i - min_val_step > params['patience'] or i == max_steps - 1:
      ckpt.restore(save_path)
      finished_training = True
      logging.info('Best validation loss was {:.3f} at step {:d}'
                   ' - stopping training'.format(min_val_loss, min_val_step))

    if i % write_freq == 0 or finished_training:
      # import pdb; pdb.set_trace()
      utils.write_losses_to_log(loss_list, range(start_printing, i + 1),
                                params['logdir'])
      start_printing = i + 1
      utils.plot_losses(params['figdir'], loss_list)
      logging.info('Step {:d}: Wrote losses and plots'.format(i))

    if finished_training:
      break


def test_classifier(clf, itr_test, params, test_name):
  """Test a trained classifier."""

  max_steps_test = params['max_steps_test']
  run_avg_len = params['run_avg_len']

  ra_loss = RALoss('loss', run_avg_len)
  ra_error = RALoss('error', run_avg_len)

  loss_tensor = []
  err_tensor = []
  label_tensor = []
  preds_tensor = []
  reprs_collection = [list() for l in range(params['n_layers'])]
  for i in range(max_steps_test):
    batch_x, batch_y = itr_test.next()
    loss, error, preds, reprs = clf.get_loss(batch_x,
                                             batch_y, return_preds=True)

    losses = list(zip([ra_loss, ra_error],
                 [tf.reduce_mean(l) for l in
                  (loss, error)]))
    utils.update_losses(losses)
    utils.print_losses([l[0] for l in losses], i)
    loss_tensor.append(loss)
    err_tensor.append(error)
    preds_tensor.append(preds)
    label_tensor.append(batch_y)
    for l in range(params['n_layers']):
      reprs_collection[l].append(reprs[l])

  loss_tensor = tf.concat(loss_tensor, 0)
  err_tensor = tf.concat(err_tensor, 0)
  preds_tensor = tf.concat(preds_tensor, 0)
  label_tensor = tf.concat(label_tensor, 0)
  for i in range(params['n_layers']):
    reprs_collection[i] = tf.concat(reprs_collection[i], 0)
  utils.save_tensors(zip(
      [ra_loss, ra_error, RALoss('preds', 1), RALoss('labels', 1)],
      [loss_tensor, err_tensor, preds_tensor, label_tensor]),
                     params['tensordir'])
  utils.save_tensors(zip(
      [RALoss('repr_{:d}'.format(l), 1) for l in range(params['n_layers'])],
      [r[:params['n_test_save']] for r in reprs_collection]), params['tensordir'])
  utils.write_metrics(zip(['loss', 'error'],
                          [np.mean(loss_tensor), np.mean(err_tensor)]),
                      params['logdir'], test_name)


