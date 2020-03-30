import matplotlib as mpl
import matplotlib.pyplot as plt 
import numpy as np
from .utils import t2n
import os 
from sklearn.decomposition import PCA

EPS = np.finfo(float).eps 

class _Plot:
    def __init__(self, config):
        self.config = config 
        self.reset()
        if not os.path.exists(config.plot):
            os.makedirs(config.plot)

    @staticmethod
    def _get_range(x, n_std=2):
        mean = np.mean(x)
        width = n_std * np.std(x)
        return mean - width, mean + width


class RecoPlot(_Plot):
    def add_values(self, x, xhat):
        self.xs.append(t2n(x))
        self.xhats.append(t2n(xhat))

    def plot(self, suffix=''):
        x = np.concatenate(self.xs, axis=0)
        xhat = np.concatenate(self.xhats, axis=0)

        mask = (x > 0).sum(axis=1).astype(bool)
        x = x[mask, :]
        xhat = xhat[mask, :]

        for i in range(x.shape[1]):
            plt.clf()
            ranges = (self._get_range(x, 1), self._get_range(xhat, 1))
            plt.hist2d(x[:,i], xhat[:,i], range=ranges, bins=50, cmin=1,
                       norm=mpl.colors.LogNorm())
            plt.colorbar()
            for ext in ['pdf', 'png']:
                plt.savefig(f'{self.config.plot}/feature_{i}{suffix}.{ext}')

            plt.clf()
            err = (x[:,i] - xhat[:,i]) 
            plt.hist(err, bins=50, range=self._get_range(err, 3))
            for ext in ['pdf', 'png']:
                plt.savefig(f'{self.config.plot}/error_{i}{suffix}.{ext}')
        self.reset()

    def reset(self):
        self.xs, self.xhats = [], []


class LatentPlot(_Plot):
    def reset(self):
        self.latents = []

    def add_values(self, latent):
        self.latents.append(t2n(latent))

    def plot(self, suffix=''):
        x = np.concatenate(self.latents, axis=0)

        if x.shape[1] > 2:
            x = PCA(n_components=2).fit_transform(x)

        plt.clf() 
        ranges = (self._get_range(x[:,0]), self._get_range(x[:,1]))
        plt.hist2d(x[:,0], x[:,1], range=ranges, bins=50, cmin=1,
                   norm=mpl.colors.LogNorm())
        plt.colorbar()
        for ext in ['pdf', 'png']:
            plt.savefig(f'{self.config.plot}/latent_{suffix}.{ext}')
        self.reset()
