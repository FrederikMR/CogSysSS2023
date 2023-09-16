#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 10:54:37 2023

@author: fmry
"""

#%% Sources

"""
Sources:
https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
http://adamlineberry.ai/vae-series/vae-code-experiments
https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa
https://discuss.pytorch.org/t/cpu-ram-usage-increasing-for-every-epoch/24475/10
https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
"""

#%% Modules

#Loading own module from parent folder
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.realpath(currentdir))
parentdir = os.path.dirname(os.path.realpath(parentdir))
sys.path.append(parentdir)

#Modules
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import argparse
import pandas as pd
import numpy as np
import datetime

#Own files
from VAE.Surface3D import VAE_3D

#%% Parser for command line arguments

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--data_path', default='../Data/circle/circle.csv', # 'Data/surface_R2.csv'
                        type=str)
    parser.add_argument('--save_model_path', default='models/circle/vae_circle', #'trained_models/surface_R2'
                        type=str)
    parser.add_argument('--model_number', default='_1', #'trained_models/surface_R2'
                        type=str)
    parser.add_argument('--save_hours', default=1,
                        type=float)

    #Hyper-parameters
    parser.add_argument('--device', default='cpu', #'cuda:0'
                        type=str)
    parser.add_argument('--epochs', default=100, #100000
                        type=int)
    parser.add_argument('--batch_size', default=100,
                        type=int)
    parser.add_argument('--lr', default=0.0001,
                        type=float)
    parser.add_argument('--workers', default=0,
                        type=int)
    parser.add_argument('--latent_dim', default=2,
                        type=int)
                       

    #Continue training or not
    parser.add_argument('--con_training', default=0,
                        type=int)
    parser.add_argument('--load_model_path', default='models/circle/vae_circle.pt',
                        type=str)


    args = parser.parse_args()
    return args

#%% Sim circle

def data_generator(N):
    
    args = parse_args()
    
    mu = np.array([1.0, 1.0, 1.0])
    
    theta = np.random.uniform(0, 2*np.pi, N)
    x1 = np.cos(theta)+mu[0]
    x2 = np.sin(theta)+mu[1]
    x3 = np.zeros(N)+mu[2]
    
    df = np.vstack((x1,x2,x3))
    
    pd.DataFrame(df).to_csv(args.data_path)
    
    return

#%% Main loop

def main():

    args = parse_args()
    train_loss_elbo = [] #Elbo loss
    train_loss_rec = [] #Reconstruction loss
    train_loss_kld = [] #KLD loss
    epochs = args.epochs
    time_diff = datetime.timedelta(hours=args.save_hours)
    start_time = datetime.datetime.now()
    current_time = start_time
    
    if not os.path.isfile(args.data_path):
        data_generator(50000)

    df = pd.read_csv(args.data_path, index_col=0)
    DATA = torch.Tensor(df.values).to(args.device) #DATA = torch.Tensor(df.values)
    DATA = torch.transpose(DATA, 0, 1)

    if args.device == 'cpu':
        trainloader = DataLoader(dataset = DATA, batch_size= args.batch_size,
                                 shuffle = True, pin_memory=True, num_workers = args.workers)
    else:
        trainloader = DataLoader(dataset = DATA, batch_size= args.batch_size,
                                 shuffle = True)
       
    N = len(trainloader.dataset)

    model = VAE_3D(fc_h = [3, 100],
                 fc_g = [args.latent_dim, 100, 3],
                 fc_mu = [100, args.latent_dim],
                 fc_var = [100, args.latent_dim],
                 fc_h_act = [nn.ELU],
                 fc_g_act = [nn.ELU, nn.Identity],
                 fc_mu_act = [nn.Identity],
                 fc_var_act = [nn.Sigmoid]).to(args.device) #Model used

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.con_training:
        checkpoint = torch.load(args.load_model_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        elbo = checkpoint['ELBO']
        rec_loss = checkpoint['rec_loss']
        kld_loss = checkpoint['KLD']

        train_loss_elbo = elbo
        train_loss_rec = rec_loss
        train_loss_kld = kld_loss
    else:
        last_epoch = 0

    model.train()
    for epoch in range(last_epoch, epochs):
        running_loss_elbo = 0.0
        running_loss_rec = 0.0
        running_loss_kld = 0.0
        for x in trainloader:
            #x = x.to(args.device) #If DATA is not saved to device
            _, x_hat, mu, var, kld, rec_loss, elbo = model(x)
            #optimizer.zero_grad(set_to_none=True) #Based on performance tuning
            optimizer.zero_grad()
            elbo.backward()
            optimizer.step()

            running_loss_elbo += elbo.item()
            running_loss_rec += rec_loss.item()
            running_loss_kld += kld.item()

            #del x, x_hat, mu, var, kld, rec_loss, elbo #In case you run out of memory

        train_epoch_loss = running_loss_elbo/N
        train_loss_elbo.append(train_epoch_loss)
        train_loss_rec.append(running_loss_rec/N)
        train_loss_kld.append(running_loss_kld/N)

        current_time = datetime.datetime.now()
        if current_time - start_time >= time_diff:
            print(f"Epoch {epoch+1}/{epochs} - loss: {train_epoch_loss:.4f}")
            checkpoint = args.save_model_path+args.model_number+'.pt'
            torch.save({'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ELBO': train_loss_elbo,
                'rec_loss': train_loss_rec,
                'KLD': train_loss_kld
                }, checkpoint)
            start_time = current_time


    checkpoint = args.save_model_path+args.model_number+'.pt'
    torch.save({'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ELBO': train_loss_elbo,
                'rec_loss': train_loss_rec,
                'KLD': train_loss_kld
                }, checkpoint)

    return

#%% Calling main

if __name__ == '__main__':
    main()