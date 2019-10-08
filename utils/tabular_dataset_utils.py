import os
import numpy as np
import sklearn
import sklearn.datasets
from utils.dataset_iterator import DatasetIterator

DATADIR = '/scratch/gobi1/madras/datasets'

def load_boston():
    dat = sklearn.datasets.load_boston()
    x, y = dat['data'], np.expand_dims(dat['target'], 1)
    return x, y

def load_diabetes():
    dat = sklearn.datasets.load_diabetes()
    x, y = dat['data'], np.expand_dims(dat['target'], 1)
    return x, y

def load_wine():
    d = np.load(os.path.join(DATADIR, 'winequality', 'winequality.npz'))
    return d['x'], d['y']

def load_abalone():
    d = np.load(os.path.join(DATADIR, 'abalone', 'abalone.npz'))
    return d['x'], d['y']

def make_ood_split(x, y, ood_identifier):
    '''ood_identifier should be a string in {'none', 'sim_feature-d-n'},
    where d is an integer representing the number of features to simulate
    and n is a float representing the stdev of the added noise distribution.'''

    pct_ood = 0.3
    if ood_identifier == 'none':
        ood_indicator = np.random.binomial(1, pct_ood, size=x.shape[0])
        oodist_inds = ood_indicator.flatten().astype(bool)
        indist_inds = np.logical_not(oodist_inds)
        indist_x, indist_y = x[indist_inds], y[indist_inds]
        oodist_x, oodist_y = x[oodist_inds], y[oodist_inds]

    elif ood_identifier.startswith('sim_feature'):
        x -= np.mean(x, axis=0, keepdims=True)
        x /= np.std(x, axis=0, keepdims=True)

        ood_indicator = np.random.binomial(1, pct_ood, size=x.shape[0])
        oodist_inds = ood_indicator.flatten().astype(bool)
        indist_inds = np.logical_not(oodist_inds)
        indist_x, indist_y = x[indist_inds], y[indist_inds]
        oodist_x, oodist_y = x[oodist_inds], y[oodist_inds]
        
        numfeat = int(ood_identifier.split('-')[1])
        for _ in range(numfeat):
            i1 = np.random.randint(x.shape[1])
            i2 = np.random.randint(x.shape[1])

            coeff = np.random.rand()
            indist_x_new_feature = (coeff * indist_x[:, i1: i1 + 1] + 
                                    (1 - coeff) * indist_x[:, i2: i2 + 1])
            oodist_x_new_feature = np.random.choice(
                    indist_x_new_feature.flatten(), size=(oodist_x.shape[0], 1))

            indist_x = np.concatenate([indist_x, indist_x_new_feature], axis=1)
            oodist_x = np.concatenate([oodist_x, oodist_x_new_feature], axis=1)

    else:
        raise Exception('ood identifier is {}, not implemented'
                        .format(ood_identifier))
    return indist_x, indist_y, oodist_x, oodist_y 

def load_regression_dataset(ood_identifier, dataset_name, n_batch=32, 
                            normalize=True, seed=None):
    np.random.seed(seed)
    pct_test = 0.2
    pct_valid = 0.2

    if dataset_name == 'boston':
        x, y = load_boston()
    elif dataset_name == 'diabetes':
        x, y = load_diabetes()
    elif dataset_name == 'wine':
        x, y = load_wine()
    elif dataset_name == 'abalone':
        x, y = load_abalone()
    else:
        raise Exception('name is an unsupported dataset name: {}'.format(name))

    indist_x, indist_y, oodist_x, oodist_y = make_ood_split(
                                                        x, y, ood_identifier)
    if normalize:
        indist_x_mean = np.mean(indist_x, axis=0, keepdims=True)
        indist_x_std = np.std(indist_x, axis=0, keepdims=True)
        indist_x -= indist_x_mean
        indist_x /= indist_x_std
        oodist_x -= indist_x_mean
        oodist_x /= indist_x_std

    if ood_identifier.startswith('sim_feature'):
        numfeat = int(ood_identifier.split('-')[1])
        if numfeat > 0:
            stdev = float(ood_identifier.split('-')[2])
            indist_noise = np.random.normal(0., stdev, 
                                            size=(indist_x.shape[0], numfeat))
            oodist_noise = np.random.normal(0., stdev, 
                                            size=(oodist_x.shape[0], numfeat))
            indist_x[:,-numfeat:] += indist_noise
            oodist_x[:,-numfeat:] += oodist_noise

    n_indist = indist_x.shape[0]
    n_tr = int(n_indist * (1 - pct_test))
    n_va = int(n_indist * pct_valid)
    n_te = n_indist - n_tr
    n_oo = oodist_x.shape[0]
    
    train_inds = np.random.choice(range(n_indist), size=n_tr) 
    test_inds = list(filter(lambda n: not n in train_inds, range(n_indist)))

    valid_inds = np.random.choice(train_inds, size=n_va) 
    train_inds = list(filter(lambda n: not n in valid_inds, train_inds))
    train_x, train_y = indist_x[train_inds], indist_y[train_inds]
    valid_x, valid_y = indist_x[valid_inds], indist_y[valid_inds]
    test_x, test_y = indist_x[test_inds], indist_y[test_inds]
    
    print('Train/Valid/Test/OOD: {:d}/{:d}/{:d}/{:d}'
            .format(n_tr, n_va, n_te, n_oo))
    
    itr_train = DatasetIterator(train_x, train_y, n_batch)
    itr_valid = DatasetIterator(valid_x, valid_y, n_batch)
    itr_test = DatasetIterator(test_x, test_y, n_batch)
    itr_ood = DatasetIterator(oodist_x, oodist_y, n_batch)
    
    return itr_train, itr_valid, itr_test, itr_ood

