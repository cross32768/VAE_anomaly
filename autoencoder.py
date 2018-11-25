# module for parts of autoencoder

import torch
from torch import nn

class Downsampler(nn.Module):
    def __init__(self, in_channels, out_channels, ksize = 4, ssize = 2, psize = 1):
        super(Downsampler, self).__init__()
        self.cv = nn.Conv2d(in_channels, out_channels, kernel_size = ksize, stride = ssize, padding = psize)
        self.bn = nn.BatchNorm2d(out_channels)
        self.rl = nn.ReLU()
        
    def only_Conv2d(self, x):
        return self.cv(x)
        
    def forward(self, x):
        out = self.cv(x)
        out = self.bn(out)
        out = self.rl(out)
        return out

class Upsampler(nn.Module):
    def __init__(self, in_channels, out_channels, ksize = 4, ssize = 2, psize = 1):
        super(Upsampler, self).__init__()
        self.tc = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = ksize, stride = ssize, padding = psize)
        self.bn = nn.BatchNorm2d(out_channels)
        self.rl = nn.ReLU()
        
    def only_ConvTranspose2d(self, x):
        return self.tc(x)
        
    def forward(self, x):
        out = self.tc(x)
        out = self.bn(out)
        out = self.rl(out)
        return out

class AutoEncoder(nn.Module):
    def __init__(self, n):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
                            Downsampler(   3,   n),
                            Downsampler(   n, 2*n),
                            Downsampler( 2*n, 4*n),
                            Downsampler( 4*n, 8*n)
         )
        
        self.decoder = nn.Sequential(
                            Upsampler( 8*n, 4*n),
                            Upsampler( 4*n, 2*n),
                            Upsampler( 2*n,   n),
                            nn.ConvTranspose2d( n, 3, kernel_size = 4, stride = 2, padding = 1)
        )
        
    def get_feature_map(self, x, n_layers):
        if n_layers > len(self.encoder) or n_layers < 1:
            raise ValueError('n_layres must be in [1,2,...len(self_encoder)]')
            
        for i in range(n_layers-1):
            x = self.encoder[i](x)
        out = self.encoder[n_layers-1].only_Conv2d(x)
        return out
        
    def forward(self, x):
        encode_x = self.encoder(x)
        recon_x = torch.tanh(self.decoder(encode_x))
        return recon_x

