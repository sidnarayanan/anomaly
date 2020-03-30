import torch
from torch import nn 


class AE(nn.Module):
    _latent_factor = 1

    def __init__(self, args):
        super().__init__()

        self.args = args

        encoder_sizes = [args.n_features] + args.n_hidden + [args.n_latent * self._latent_factor]
        decoder_sizes = [args.n_latent] + args.n_hidden[::-1] + [args.n_features]

        self.encoder_layers = nn.ModuleList([
            nn.Linear(i, o) for i, o in zip(encoder_sizes[:-1], encoder_sizes[1:])
        ])
        self.decoder_layers = nn.ModuleList([
            nn.Linear(i, o) for i, o in zip(decoder_sizes[:-1], decoder_sizes[1:])
        ])
        self.act = nn.ReLU() # defining this up here makes it show up in torchsummary

    def encode(self, x):
        h = x
        for l in self.encoder_layers[:-1]:
            h = l(h)
            h = self.act(h)
        h = self.encoder_layers[-1](h)
        return h

    def decode(self, x):
        h = x
        for l in self.decoder_layers[:-1]:
            h = l(h)
            h = self.act(h)
        h = self.decoder_layers[-1](h)
        return h

    def forward(self, x, mask=None):
        h = self.encode(x)
        h = self.decode(h)
        return h 


class VAE(AE):
    _latent_factor = 2

    def sample(self, x):
        n_latent = self.args.n_latent 
        mu, logvar = x[:,:n_latent], x[:,n_latent:]
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def forward(self, x, mask=None):
        h = self.encode(x)
        h = self.sample(h)
        h = self.decode(h)
        return h 
