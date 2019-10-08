import numpy as np
from sklearn.metrics import roc_auc_score

def get_auc(test_score, ood_score):
    is_ood_indicator = np.concatenate([np.zeros_like(test_score), np.ones_like(ood_score)], axis=0)
    test_and_ood_scores = np.concatenate([test_score, ood_score], axis=0)
    auc = roc_auc_score(is_ood_indicator, test_and_ood_scores)
    return auc

def get_min_dist(x, trdata):
    # x is (1 by d), trdata is (n by d)
    axis = (1, 2) if x.ndim > 2 else 1
    dists = np.linalg.norm(trdata - x, axis=axis)
    min_dist = np.amin(dists)
    return min_dist

# for each example in reprs, find distance to NN in valid reprs
def get_all_min_dists(trdata, testdata):
    min_dists = []
    for i in range(testdata.shape[0]):
        x = testdata[i: i + 1]
        min_dist = get_min_dist(x, trdata)
        min_dists.append(min_dist)
    min_dists = np.array(min_dists)
    return min_dists
