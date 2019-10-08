import numpy as np
import tensorflow as tf
import local_ensembles.evaluation_functions as eval_fns
from utils import tensor_utils as tu

def get_preds_and_reprs_in_minibatches(model, x, mb_size):
    """Gets predictions and hidden layer representations for a batch of
        examples x with respect to a model.

    Args:
        x (tensor): the examples we wish to get the predictions and
            representations for.
        model (tf.keras.Model): a pretrained model.
        mb_size (int): how many examples of x to feed through model at once.
    Returns:
        preds (tensor): predictions for each element in x. If x
            is n x d and the model outputs space is l, preds is n x l.
        reprs (list of tensors): model hidden representations for each
            element in x. If model has m layers, and layer i has p_i parameters
            in it, then the i-th element of reprs has shape n x p_i.
    """
    curr = 0
    num_examples = x.shape[0]
    preds = []
    reprs = {}
    while curr < num_examples:
        batch_x = x[curr: min(curr + mb_size, num_examples)]
        batch_preds, batch_reprs = model(batch_x)
        preds.append(batch_preds)
        for i, batch_repr in enumerate(batch_reprs):
            if not i in reprs:
                reprs[i] = []
            reprs[i].append(batch_repr)
        curr += mb_size
    preds = tf.concat(preds, axis=0)
    for i in reprs:
        reprs[i] = tf.concat(reprs[i], axis=0)
    return preds, [reprs[i] for i in range(len(reprs))]

def run_baselines(model, itr_train, itr_test, itr_ood, mb_size):
    """Run baselines methods for OOD detection.

    Args:
        model (tf.keras.Model): a model we are examining.
        itr_train (Iterator): training data iterator.
        itr_test (Iterator): testing data iterator.
        itr_ood (Iterator): OOD data iterator.
        mb_size (int): size of minibatches to use.
    Returns:
        res (dict): dictionary of baselines AUC results, for methods:
            Maxprob: maximum outputted softmax probability.
            NN-Pixels: nearest neighbour in pixel space.
            NN-Reprs: nearest neighbour in representation space.
            NN-Repr-Final: nearest neighbour in final hidden layer space.
    """
    # run baselines
    _, train_reprs_tf = get_preds_and_reprs_in_minibatches(
        model, itr_train.x, mb_size)
    test_preds, test_reprs_tf = get_preds_and_reprs_in_minibatches(
        model, itr_test.x, mb_size)
    ood_preds, ood_reprs_tf = get_preds_and_reprs_in_minibatches(
        model, itr_ood.x, mb_size)

    train_reprs_last = train_reprs_tf[-2].numpy()
    train_reprs_flat_tf = tu.flat_concat(train_reprs_tf)
    del train_reprs_tf
    train_reprs_flat = train_reprs_flat_tf.numpy()
    del train_reprs_flat_tf

    test_reprs_last = test_reprs_tf[-2].numpy()
    test_reprs_flat_tf = tu.flat_concat(test_reprs_tf)
    del test_reprs_tf
    test_reprs_flat = test_reprs_flat_tf.numpy()
    del test_reprs_flat_tf

    ood_reprs_last = ood_reprs_tf[-2].numpy()
    ood_reprs_flat_tf = tu.flat_concat(ood_reprs_tf)
    del ood_reprs_tf
    ood_reprs_flat = ood_reprs_flat_tf.numpy()
    del ood_reprs_flat_tf

    # MaxProb baseline
    test_probs = tf.nn.softmax(test_preds)
    ood_probs = tf.nn.softmax(ood_preds)
    auc_maxprobs = eval_fns.get_auc(-np.amax(test_probs, axis=1),
                                    -np.amax(ood_probs, axis=1))
    print('Maxprob AUC = {:.2f}'.format(auc_maxprobs))

    # Now nearest-neighbour (NN) baselines.
    # NN in pixel space
    test_mindists = eval_fns.get_all_min_dists(itr_train.x, itr_test.x)
    ood_mindists = eval_fns.get_all_min_dists(itr_train.x, itr_ood.x)
    auc_mindists = eval_fns.get_auc(test_mindists, ood_mindists)
    print('NN-Pixels AUC = {:.2f}'.format(auc_mindists))

    # NN in final layer of representation space
    test_repr_final_mindists = eval_fns.get_all_min_dists(train_reprs_last,
                                                          test_reprs_last)
    ood_repr_final_mindists = eval_fns.get_all_min_dists(train_reprs_last,
                                                         ood_reprs_last)
    auc_repr_final_mindists = eval_fns.get_auc(test_repr_final_mindists,
                                               ood_repr_final_mindists)
    print('NN-Repr-Final AUC = {:.2f}'.format(auc_repr_final_mindists))

    # NN in all of representation space
    test_repr_mindists = eval_fns.get_all_min_dists(train_reprs_flat,
                                                    test_reprs_flat)
    ood_repr_mindists = eval_fns.get_all_min_dists(train_reprs_flat,
                                                   ood_reprs_flat)
    auc_repr_mindists = eval_fns.get_auc(test_repr_mindists,
                                         ood_repr_mindists)
    print('NN-Reprs AUC = {:.2f}'.format(auc_repr_mindists))

    return {'Maxprob': auc_maxprobs,
            'NN-Pixels': auc_mindists,
            'NN-Repr-Final': auc_repr_final_mindists,
            'NN-Reprs': auc_repr_mindists
           }

