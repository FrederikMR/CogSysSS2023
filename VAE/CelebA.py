#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 07:13:59 2023

@author: fmry
"""

#%% Sources

"""
Sources used:
https://github.com/bhpfelix/Variational-Autoencoder-PyTorch/blob/master/src/vanila_vae.py
https://arxiv.org/abs/1312.6114
https://github.com/sksq96/pytorch-vae/blob/master/vae.py
"""


#%% Modules

import torch
from torch import nn

#%% VAE for 3d surfaces using article network with variance and probabilities

#The training script should be modified for the version below.
class VAE_CELEBA(nn.Module):
    def __init__(self,
                 latent_dim = 32
                 ):
        super(VAE_CELEBA, self).__init__()
               
        #Encoder
        self.h_con1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 4, stride = 2, bias = False) #31x31z32
        self.h_batch1 = nn.BatchNorm2d(32)
       
        self.h_con2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 4, stride = 2, bias = False) #14x14x32
        self.h_batch2 = nn.BatchNorm2d(32)
       
        self.h_con3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2, bias = False) #6x6x64
        self.h_batch3 = nn.BatchNorm2d(64)
       
        self.h_con4 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 4, stride = 2, bias = False) #2x2x64
        self.h_batch4 = nn.BatchNorm2d(64)
       
        self.h_fc = nn.Linear(256, 256)
        self.h_batch5 = nn.BatchNorm1d(256)
       
        #Mean and std
        self.h_mean = nn.Linear(256, latent_dim)
        self.h_std = nn.Linear(256, latent_dim)
       
        #Decoder
        self.g_fc = nn.Linear(latent_dim, 256)
        self.g_batch1 = nn.BatchNorm1d(256)
       
        self.g_tcon1 = nn.ConvTranspose2d(in_channels = 256, out_channels = 64, kernel_size = 6, stride = 2)
        self.g_batch2 = nn.BatchNorm2d(64)
       
        self.g_tcon2 = nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = 4, stride = 2)
        self.g_batch3 = nn.BatchNorm2d(64)
       
        self.g_tcon3 = nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 4, stride = 2)
        self.g_batch4 = nn.BatchNorm2d(32)
       
        self.g_tcon4 = nn.ConvTranspose2d(in_channels = 32, out_channels = 32, kernel_size = 4, stride = 2)
        self.g_batch5 = nn.BatchNorm2d(32)
       
        self.g_tcon5 = nn.ConvTranspose2d(in_channels = 32, out_channels = 3, kernel_size = 3, stride = 1)
        
        nn.init.xavier_normal_(self.h_con1.weight)
        
        nn.init.xavier_normal_(self.h_con2.weight)
        
        nn.init.xavier_normal_(self.h_con3.weight)
        
        nn.init.xavier_normal_(self.h_con4.weight)
        
        nn.init.xavier_normal_(self.h_fc.weight)
        nn.init.zeros_(self.h_fc.bias)
        
        nn.init.xavier_normal_(self.h_mean.weight)
        nn.init.zeros_(self.h_mean.bias)
        
        nn.init.xavier_normal_(self.h_std.weight)
        nn.init.zeros_(self.h_std.bias)
        
        nn.init.xavier_normal_(self.g_fc.weight)
        nn.init.zeros_(self.g_fc.bias)
        
        nn.init.xavier_normal_(self.g_tcon1.weight)
        
        nn.init.xavier_normal_(self.g_tcon2.weight)
        
        nn.init.xavier_normal_(self.g_tcon3.weight)
        
        nn.init.xavier_normal_(self.g_tcon4.weight)
        
        nn.init.xavier_normal_(self.g_tcon5.weight)

        self.ELU = nn.ELU()
        self.Sigmoid = nn.Sigmoid()
       
        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
               
    def encoder(self, x):
               
        x1 = self.ELU(self.h_batch1(self.h_con1(x)))
        x2 = self.ELU(self.h_batch2(self.h_con2(x1)))
        x3 = self.ELU(self.h_batch3(self.h_con3(x2)))
        x4 = self.ELU(self.h_batch4(self.h_con4(x3)))
       
        x4 = x4.view(x4.size(0), -1)
        x5 = self.ELU(self.h_batch5(self.h_fc(x4)))
       
        mu = self.h_mean(x5)
        std = self.Sigmoid(self.h_std(x5))
       
        return mu, std
       
    def rep_par(self, mu, std):
       
        eps = torch.randn_like(std)
        z = mu + (std*eps)
        return z
       
    def decoder(self, z):
               
        x1 = self.ELU(self.g_batch1(self.g_fc(z)))
        x1 = x1.view(-1, 256, 1, 1)
       
        x2 = self.ELU(self.g_batch2(self.g_tcon1(x1)))
        x3 = self.ELU(self.g_batch3(self.g_tcon2(x2)))
        x4 = self.ELU(self.g_batch4(self.g_tcon3(x3)))
        x5 = self.ELU(self.g_batch5(self.g_tcon4(x4)))
       
        x_hat = self.g_tcon5(x5)
       
        return x_hat
   
    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
       
        return log_pxz.sum(dim=(1,2,3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
       
        return kl
       
    def forward(self, x):
       
        mu, std = self.encoder(x)
       
        z = self.rep_par(mu, std)
       
        x_hat = self.decoder(z)
       
        # compute the ELBO with and without the beta parameter:
        # `L^\beta = E_q [ log p(x|z) - \beta * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
        kld = self.kl_divergence(z, mu, std)
        rec_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)
       
        # elbo
        elbo = (kld - rec_loss)
        elbo = elbo.mean()
       
        return z, x_hat, mu, std, kld.mean(), -rec_loss.mean(), elbo
           
    def h(self, x):
       
        mu, std = self.encoder(x)
              
        return mu
       
    def g(self, z):
       
        x_hat = self.decoder(z)
       
        return x_hat