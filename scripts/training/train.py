#!/usr/bin/env python3
from anomaly.utils import * 

p = ArgumentParser()
p.add_args(
    '--train_dataset_pattern', 
    '--test_dataset_pattern', 
    '--output', ('--n_epochs', p.INT),
    '--n_hidden', ('--n_latent', p.INT), ('--n_features', p.INT),
    ('--batch_size', p.INT),
    ('--lr', {'type': float}),
    ('--lr_schedule', p.STORE_TRUE), 
    ('--variational', p.STORE_TRUE), 
    '--plot',
)
config = p.parse_args()


from anomaly.data import FeatureDataset, DataLoader
from anomaly.model import AE, VAE
from anomaly.plot import RecoPlot, LatentPlot

from tqdm import tqdm, trange
from loguru import logger
import torch
from torch import nn 
from torch.utils.data import RandomSampler
import os
import numpy as np
from torchsummary import summary


if __name__ == '__main__':
    snapshot = Snapshot(config.output, config)
    logger.info(f'Saving output to {snapshot.path}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device

    logger.info(f'Reading dataset at {config.train_dataset_pattern}')
    train_ds = FeatureDataset(config.test_dataset_pattern)
    test_ds = FeatureDataset(config.train_dataset_pattern)
    # sampler = RandomSampler(train_ds)
    train_dl = DataLoader(train_ds, batch_size=config.batch_size)
    test_dl = DataLoader(test_ds, batch_size=config.batch_size)

    logger.info(f'Building model')
    model = (VAE if config.variational else AE)(config)
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    # lr = torch.optim.lr_scheduler.ExponentialLR(opt, config.lr_decay)
    lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, 
            factor=config.lr_decay,
            patience=3
        )

    if not os.path.exists(config.plot):
        os.makedirs(config.plot)

    summary(model, input_size=(config.n_features,))

    loss_fn = torch.nn.MSELoss()

    plot = RecoPlot(config)
    lplot = LatentPlot(config)

    best_loss = np.inf

    for e in range(config.n_epochs):
        logger.info(f'Epoch {e}: Start')
        current_lr = [group['lr'] for group in opt.param_groups][0]
        logger.info(f'Epoch {e}: Current LR = {current_lr}')

        model.train()
        train_avg_loss_tensor = 0
        for n_batch, x in enumerate(tqdm(train_dl, total=int(len(train_ds)/config.batch_size), leave=False)):
            x = torch.Tensor(x).to(device)
            
            opt.zero_grad()
            xhat = model(x)
            loss = loss_fn(x, xhat) 
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()

        model.eval()
        train_avg_loss_tensor = 0
        for n_batch, x in enumerate(tqdm(train_dl, total=int(len(train_ds)/config.batch_size), leave=False)):
            x = torch.Tensor(x).to(device)
            
            with torch.no_grad():
                xhat = model(x)
                loss = loss_fn(x, xhat) 
                train_avg_loss_tensor += loss 
            plot.add_values(x, xhat)
            lplot.add_values(model.encode(x))
        train_avg_loss_tensor /= n_batch
        train_avg_loss = t2n(train_avg_loss_tensor)

        if e % 5 == 4:
            plot.plot('_train')
        lplot.plot('_train')
        plot.reset(); lplot.reset()

        model.eval()
        test_avg_loss_tensor = 0
        for n_batch, x in enumerate(tqdm(test_dl, total=int(len(test_ds)/config.batch_size), leave=False)):
            x = torch.Tensor(x).to(device)
            
            with torch.no_grad():
                xhat = model(x)
                loss = loss_fn(x, xhat) 
                test_avg_loss_tensor += loss 
                lplot.add_values(model.encode(x))
            plot.add_values(x, xhat)

        test_avg_loss_tensor /= n_batch
        test_avg_loss = t2n(test_avg_loss_tensor)
        lr.step(test_avg_loss_tensor)

        if e % 5 == 4:
            plot.plot('_test')
        lplot.plot('_test')
        plot.reset(); lplot.reset()

        # plot_path = f'{config.plot}/resolution_{e:03d}'
        # ress = met.plot(plot_path)
        # model_res = ress['model']
        # puppi_res = ress['puppi']

        # avg_loss, avg_acc, avg_posacc, avg_negacc, avg_posfrac = metrics.mean()
        # logger.info(f'Epoch {e}: Average fraction of hard particles = {avg_posfrac}')
        logger.info('')
        logger.info(f'Epoch {e}: MODEL:')
        logger.info(f'Epoch {e}: Train loss = {train_avg_loss}; test loss = {test_avg_loss}')
        # logger.info(f'Epoch {e}: Hard ID = {avg_posacc}; PU ID = {avg_negacc}')
        # logger.info(f'Epoch {e}: MET error = {model_res[0]} +/- {model_res[1]}') 
        # 
        # avg_loss, avg_acc, avg_posacc, avg_negacc, _ = metrics_puppi.mean()
        # logger.info(f'Epoch {e}: PUPPI:')
        # logger.info(f'Epoch {e}: Loss = {avg_loss}; Accuracy = {avg_acc}')
        # logger.info(f'Epoch {e}: Hard ID = {avg_posacc}; PU ID = {avg_negacc}')
        # logger.info(f'Epoch {e}: MET error = {puppi_res[0]} +/- {puppi_res[1]}') 

        torch.save(model.state_dict(), snapshot.get_path(f'model_weights_epoch{e}.pt'))
        if test_avg_loss < best_loss:
            best_loss = test_avg_loss
            torch.save(model.state_dict(), snapshot.get_path(f'model_weights_best.pt'))
