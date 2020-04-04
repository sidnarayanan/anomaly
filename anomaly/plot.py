import matplotlib as mpl
import matplotlib.pyplot as plt 
import numpy as np
from .utils import t2n
import os 
from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as TSNE
from umap import UMAP


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

    @staticmethod
    def _get_quantiles(x, q=1):
        q = (1 - q) * 0.5
        lo = np.quantile(x, q)
        hi = np.quantile(x, 1-q)
        return lo, hi


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

        dimred = PCA(n_components=2)
        #dimred = TSNE(n_components=2)
        if x.shape[1] > 2:
            x = dimred.fit_transform(x)

        plt.clf() 
        ranges = (self._get_range(x[:,0]), self._get_range(x[:,1]))
        plt.hist2d(x[:,0], x[:,1], range=ranges, bins=50, cmin=1,
                   norm=mpl.colors.LogNorm())
        plt.colorbar()
        for ext in ['pdf', 'png']:
            plt.savefig(f'{self.config.plot}/latent_{suffix}.{ext}')
        self.reset()


class MulticlassRecoPlot(_Plot):
    _colors = [getattr(mpl.cm, c) for c in ['Reds', 'Greens', 'Blues', 'Purples', 'Greys']]
    _colors_1d = ['r', 'g', 'b', 'k']
    def reset(self):
        self.errs = {} 

    def add_values(self, x, xhat, key):
        if key not in self.errs:
            self.errs[key] = []
        self.errs[key].append((t2n(x) - t2n(xhat)).mean(axis=-1))

    def plot(self, suffix=''):
        plt.clf()
        ranges = None
        for i, (k, e) in enumerate(self.errs.items()):
            err = np.concatenate(e, axis=0)

            if ranges is None:
                ranges = self._get_range(err, 3)
            plt.hist(err, bins=50, range=self._get_range(err, 3), color=self._colors_1d[i], 
                     label=k, histtype='step', density=True)
        plt.legend()
        for ext in ['pdf', 'png']:
            plt.savefig(f'{self.config.plot}/multiclass_error_{suffix}.{ext}')
        self.reset()


class MulticlassLatentPlot(_Plot):
    _colors = [getattr(mpl.cm, c) for c in ['Reds', 'Greens', 'Blues','Greys']]
    _colors_1d = ['r', 'g', 'b', 'k']
    def reset(self):
        self.latents = {} 

    def add_values(self, latent, key):
        if key not in self.latents:
            self.latents[key] = []
        self.latents[key].append(t2n(latent))

    def plot(self, suffix='', ranges=None):
        plt.clf() 
        dimred = PCA(n_components=2)
        #dimred = TSNE(n_components=2, n_jobs=-1, verbose=3)
        #dimred = UMAP(n_components=2, verbose=3)
        for i,(k,v) in enumerate(self.latents.items()):
            v = np.concatenate(v, axis=0)
            self.latents[k] = v
        all_data = np.concatenate(list(self.latents.values()), axis=0)
        all_y = np.concatenate([i*np.ones(shape=v.shape[0]) for i,v in enumerate(self.latents.values())], axis=0)
        mask = np.random.binomial(1, 0.05, size=all_y.shape).astype(bool)
        all_data, all_y = all_data[mask,:], all_y[mask]
        if all_data.shape[1] > 2:
            all_data_2 = dimred.fit_transform(all_data)

        for j in range(all_data.shape[1]):
            plt.clf()
            ranges_1d = None
            for i,(k,v) in enumerate(self.latents.items()):
                v = v[:,j]
                if ranges_1d is None:
                    ranges_1d = self._get_range(v)
                plt.hist(v, range=ranges_1d, bins=50, #cmin=1,
                         density=True,
                         color=self._colors_1d[i], #alpha=0,
                         linewidth=2,
                         histtype='step',
                         label=k
                        )
            plt.legend()
            for ext in ['pdf', 'png']:
                plt.savefig(f'{self.config.plot}/multiclass_latent_dim{j}_{suffix}.{ext}')

        alpha = 1
        plt.clf()
        self.latents_dimred = {}
        if ranges is None:
            ranges = (self._get_range(all_data_2[:,0]), self._get_range(all_data_2[:,1]))
        for i,(k,v) in enumerate(self.latents.items()):
            if v.shape[1] > 2:
                v = dimred.transform(v)
                self.latents_dimred[k] = v
            z, xe, ye = np.histogram2d(v[:,0], v[:,1], range=ranges, bins=100, density=True)
            x = (xe[:-1] + xe[1:]) * 0.5
            y = (ye[:-1] + ye[1:]) * 0.5
            plt.contour(x, y, z, 5, cmap=self._colors[i], alpha=alpha) 
            # alpha *= 0.7
        for ext in ['pdf', 'png']:
            plt.savefig(f'{self.config.plot}/multicontour_latent_{suffix}.{ext}')

        for i in range(all_data.shape[1]):
            for j in range(i):
                dimred_ij = PCA(2) 
                all_data_2 = all_data[:, [i,j]]
                all_data_2 = dimred_ij.fit_transform(all_data_2)
                ranges_ij = (self._get_range(all_data_2[:,0]), self._get_range(all_data_2[:,1]))
                plt.clf()
                alpha=1
                for c,(k,v) in zip(self._colors, self.latents.items()):
                    v = v[:,[i,j]]
                    v = dimred_ij.transform(v)
                    x, y = v[:,0], v[:,1]
                    z, xe, ye = np.histogram2d(x, y, range=ranges_ij, bins=100, density=True)
                    x = (xe[:-1] + xe[1:]) * 0.5
                    y = (ye[:-1] + ye[1:]) * 0.5
                    plt.contour(x, y, z, 5, cmap=c, alpha=alpha) 
                    # alpha *= 0.7
                for ext in ['pdf', 'png']:
                    plt.savefig(f'{self.config.plot}/multicontour_latent_d{i}{j}{suffix}.{ext}')

        '''
        alpha = 1
        plt.clf()
        self.latents_dimred = {}
        for i,(k,v) in enumerate(self.latents.items()):
            if v.shape[1] > 2:
                v = dimred.transform(v)
                self.latents_dimred[k] = v
            if ranges is None:
                ranges = (self._get_range(v[:,0]), self._get_range(v[:,1]))
            plt.hist2d(v[:,0], v[:,1], range=ranges, bins=50, #cmin=1,
                       # norm=mpl.colors.LogNorm(),
                       density=True,
                       cmap=self._colors[i], alpha=alpha,
                       label=k
                    )
            alpha *= 0.7
        for ext in ['pdf', 'png']:
            plt.savefig(f'{self.config.plot}/multiclass_latent_{suffix}.{ext}')

        for i in range(all_data.shape[1]):
            for j in range(i):
                ranges_ij = None
                plt.clf()
                alpha=1
                for c,(k,v) in zip(self._colors, self.latents.items()):
                    x, y = v[:,i], v[:,j]
                    if ranges_ij is None:
                        ranges_ij = (self._get_range(x, 2), self._get_range(y, 2))
                    plt.hist2d(x, y, range=ranges_ij, bins=50,
                               # norm=mpl.colors.LogNorm(),
                               density=True,
                               cmap=c, alpha=alpha,
                               label=k)
                    alpha *= 0.7
                for ext in ['pdf', 'png']:
                    plt.savefig(f'{self.config.plot}/multiclass_latent_d{i}{j}{suffix}.{ext}')

        for i in range(all_data.shape[1]):
            for j in range(i):
                ranges_ij = None
                plt.clf()
                for c,(k,v) in zip(self._colors, self.latents.items()):
                    x, y = v[:,i], v[:,j]
                    if ranges_ij is None:
                        ranges_ij = (self._get_range(x, 2), self._get_range(y, 2))
                    plt.hist2d(x, y, range=ranges_ij, bins=50,
                               # norm=mpl.colors.LogNorm(),
                               density=True,
                               cmap=self._colors[-1], #alpha=alpha,
                               label=k)
                    for ext in ['pdf', 'png']:
                        plt.savefig(f'{self.config.plot}/multiclass_latent_d{i}{j}_{k}{suffix}.{ext}')

        for i,(k,v) in enumerate(self.latents_dimred.items()):
            plt.clf()
            plt.hist2d(v[:,0], v[:,1], range=ranges, bins=50, #cmin=1,
                       # norm=mpl.colors.LogNorm(),
                       density=True,
                       cmap=self._colors[-1], 
                       label=k
                    )
            for ext in ['pdf', 'png']:
                plt.savefig(f'{self.config.plot}/multiclass_latent_{k}_{suffix}.{ext}')
        '''
        self.reset()
