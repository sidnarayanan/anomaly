import torch
from torch.utils.data import DataLoader, IterableDataset, Dataset
from loguru import logger
from glob import glob 
import numpy as np
from tqdm import tqdm


class FeatureDataset(IterableDataset):
    def __init__(self, pattern, mu=None, sigma=None):
        self._files = glob(pattern)
        self._len = sum([np.load(f)['x'].shape[0] for f in self._files])
        # data = np.load(self._files[0])
        # X = data['x'].astype(np.float32)
        # print(np.sum(np.isnan(X)) / np.sum(np.ones_like(X)))
        # X[np.isnan(X)] = 0
        # self.X = X
        self.mu, self.sigma = mu, sigma 

    def __len__(self):
        return self._len

    def __iter__(self):
        np.random.shuffle(self._files)
        for f in self._files:
            data = np.load(f)
            X = data['x'].astype(np.float32)
            bad_evts = np.isnan(X).sum(axis=-1).astype(bool)
            if bad_evts.astype(int).sum() > 0:
                logger.debug(f'{f} {bad_evts.sum()}')
            X = X[~bad_evts, :]
            if self.mu is None:
                self.mu = np.mean(X, axis=0)
                self.sigma = np.std(X, axis=0)
            X = (X - self.mu) / self.sigma
            idx = np.arange(X.shape[0])
            np.random.shuffle(idx)
            for i in idx:
                yield X[i, :]

    # @staticmethod
    # def collate_fn(samples):
    #     n_fts = len(samples[0])
    #     to_ret = [np.stack([s[i] for s in samples], axis=0) for i in range(n_fts)]
    #     return to_ret
