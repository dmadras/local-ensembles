import numpy as np

class DatasetIterator(object):

    def __init__(self, x, y, n_batch):
        self.x = x
        self.y = y
        self.n_data = x.shape[0]
        self.n_batch = n_batch

    def __iter__(self):
        return self

    def __next__(self):
        batch_inds = np.random.choice(range(self.n_data), size=self.n_batch)
        batch_x = self.x[batch_inds]
        batch_y = self.y[batch_inds]
        return batch_x, batch_y

    def next(self):
        return self.__next__()
