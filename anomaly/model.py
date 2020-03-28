import torch
from torch import nn 


class AutoEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        encoder_sizes = [args.n_features] + args.n_hidden + [args.n_latent]
        decoder_sizes = encoder_sizes[::-1]

        self.encoder_layers = nn.ModuleList([
            nn.Linear(i, o) for i, o in zip(encoder_sizes[:-1], encoder_sizes[1:])
        ])
        self.decoder_layers = nn.ModuleList([
            nn.Linear(i, o) for i, o in zip(decoder_sizes[:-1], decoder_sizes[1:])
        ])

    def encode(self, x):
        h = x
        for l in self.encoder_layers:
            h = l(h)
            h = torch.tanh(h)
        return h

    def decode(self, x):
        h = x
        for l in self.decoder_layers[:-1]:
            h = l(h)
            h = torch.tanh(h)
        h = self.decoder_layers[-1](h)
        return h

    def forward(self, x, mask=None):
        h = self.encode(x)
        h = self.decode(h)
        return h 
