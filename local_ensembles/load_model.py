import os
from model import classifier, regressor
from utils import utils

def load_model(expdir, model_type, ckpt_name='bestmodel'):
    """Load a pre-trained model.

    Args:
        expdir (str): directory where model checkpoint is saved.
        model_type (str): either "CNN_classifier" or "MLP_regressor", depending
                  on what type of model we wish to load.
        ckpt_name (str, optional): identifier of model checkpoint we wish to
                  load.
    Returns:
        model (tf.keras.Model): a pre-trained model.
    """
    param_file = os.path.join(expdir, 'params.json')
    model_params = utils.load_json(param_file)
    ckpt_path = os.path.join(expdir, 'ckpts/{}-1'.format(ckpt_name))
    if model_type == 'CNN_classifier':
        return load_cnn_classifier(model_params, ckpt_path)
    else:
        return load_mlp_regressor(model_params, ckpt_path)

def load_cnn_classifier(model_params, ckpt_path):
    """Load a pre-trained CNN classifier.

    Args:
        model_params (dict): dictionary of parameters defining the CNN, for
            instance as saved by train_classifier_ood.py in a params.json file
        ckpt_path (str): path where the checkpoint is saved.
    Returns:
        model (classifier.CNN): a pre-trained CNN classifier.
    """
    cnn_args = {'conv_dims':
                    [int(x) for x in model_params['conv_dims'].split(',')],
                'conv_sizes':
                    [int(x) for x in model_params['conv_sizes'].split(',')],
                'dense_sizes':
                    [int(x) for x in model_params['dense_sizes'].split(',')],
                'n_classes': model_params['n_classes'], 'onehot': True}
    model = utils.load_model(ckpt_path, classifier.CNN, cnn_args)
    return model


def load_mlp_regressor(model_params, ckpt_path):
    """Load a pre-trained MLP regression model.

    Args:
        model_params (dict): dictionary of parameters defining the model, for
            instance as saved by train_regressor_ood.py in a params.json file
        ckpt_path (str): path where the checkpoint is saved.
    Returns:
        model (regressor.MLP): a pre-trained MLP regression model.
    """
    cnn_args = {'dense_sizes':
                [int(x) for x in model_params['dense_sizes'].split(',')],
                'activation': model_params['activation']}
    model = utils.load_model(ckpt_path, regressor.MLP, cnn_args)
    return model

