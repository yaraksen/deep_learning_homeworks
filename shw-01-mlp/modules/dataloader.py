import numpy as np


class DataLoader(object):
    """
    Tool for shuffling data and forming mini-batches
    """

    def __init__(self, X, y, batch_size=1, shuffle=False):
        """
        :param X: dataset features
        :param y: dataset targets
        :param batch_size: size of mini-batch to form
        :param shuffle: whether to shuffle dataset
        """
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_id = 0  # use in __next__, reset in __iter__

    def __len__(self) -> int:
        """
        :return: number of batches per epoch
        """
        if self.X.shape[0] % self.batch_size == 0:
            return self.X.shape[0] // self.batch_size
        else:
            return self.X.shape[0] // self.batch_size + 1

    def num_samples(self) -> int:
        """
        :return: number of data samples
        """
        return self.X.shape[0]

    def __iter__(self):
        """
        Shuffle data samples if required
        :return: self
        """
        self.batch_id = 0
        if self.shuffle:
            perm = np.random.permutation(self.num_samples())
            self.X = self.X[perm]
            self.y = self.y[perm]
        return self

    def __next__(self):
        """
        Form and return next data batch
        :return: (x_batch, y_batch)
        """
        if self.batch_id < self.__len__():
            self.batch_id += 1
            return (self.X[(self.batch_id - 1) * self.batch_size: self.batch_id * self.batch_size],
                    self.y[(self.batch_id - 1) * self.batch_size: self.batch_id * self.batch_size])
        else:
            raise StopIteration
