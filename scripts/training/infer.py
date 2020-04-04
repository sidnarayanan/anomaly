#!/usr/bin/env python3
from anomaly.utils import * 

p = ArgumentParser()
p.add_args(
    ('--test_dataset_patterns', p.MANY), 
    '--output', 
    '--n_hidden', ('--n_latent', p.INT), ('--n_features', p.INT),
    ('--batch_size', p.INT),
    ('--variational', p.STORE_TRUE), 
    '--plot',
)
config = p.parse_args()


from anomaly.data import FeatureDataset, DataLoader
from anomaly.model import AE, VAE
from anomaly.plot import * 

from tqdm import tqdm, trange
from loguru import logger
import torch
from torch import nn 
from torch.utils.data import RandomSampler
import os
import numpy as np
from torchsummary import summary


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device

    logger.info(f'Reading dataset at {config.test_dataset_patterns}')
    stats = torch.load(f'{config.output}/feature_stats.pt')
    mu, sigma = stats['mu'], stats['sigma']
    test_dss = [FeatureDataset(p, mu, sigma) for p in config.test_dataset_patterns]
    test_dls = [DataLoader(test_ds, batch_size=config.batch_size) for test_ds in test_dss]

    logger.info(f'Building model')
    model = (VAE if config.variational else AE)(config)
    model = model.to(device)
    model.load_state_dict(torch.load(f'{config.output}/model_weights_best.pt'))

    if not os.path.exists(config.plot):
        os.makedirs(config.plot)

    summary(model, input_size=(config.n_features,))
    plot = MulticlassLatentPlot(config)
    rplot = MulticlassRecoPlot(config)

    model.eval()
    for pattern, dl, ds in zip(config.test_dataset_patterns, test_dls, test_dss):
        label = pattern.split('/')[-1].split('_')[0]
        for n_batch, x in enumerate(tqdm(dl, total=int(len(ds)/config.batch_size), leave=False)):
            x = torch.Tensor(x).to(device)
            with torch.no_grad():
                xhat = model(x)
                plot.add_values(model.encode(x), label)
                rplot.add_values(x, xhat, label)

    plot.plot(ranges=((-50, 100), (-10, 10)))
    rplot.plot()
