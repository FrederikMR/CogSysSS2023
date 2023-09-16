#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 21:37:26 2023

@author: fmry
"""

#%%

#Modules used

#Pytorch modules
import torch
from torch import nn
from torch import Tensor
import torch.optim as optim

#Other
from typing import Callable, Tuple, List
import datetime

class rm_generative_models(object):
   
    def __init__(self,
                 model_encoder:Callable[[Tensor], Tensor],
                 model_decoder:Callable[[Tensor], Tensor],
                 device:str = 'cpu',
                 T:int = 10,
                 save_path:str = 'model_save.pt',
                 save_time:int = 1, #Hours
                 epochs:int = 100000,
                 lr:float = 1e-3,
                 eps:float = 1e-6):
       
        self.h = model_encoder
        self.g = model_decoder
        self.T = T
        self.save_path = save_path
        self.save_time = save_time
        self.epochs = epochs
        self.lr = lr
        self.eps = eps
       
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device
   
    def linear_inter(self, z0:Tensor, zT:Tensor , T:int = None)->Tensor:
       
        if T is None:
            T = self.T
         
        z0 = z0.view(-1)
        zT = zT.view(-1)  
         
        if z0.shape != zT.shape:
            raise ValueError('z0 and zT have different shapes!')
       
        step = (zT-z0)/T
        Z = (torch.ones([T+1, z0.shape[-1]],device=self.device)*step).transpose(0,1)*torch.tensor(range(0,T+1), device=self.device)
        Z = Z.transpose(0,1)+z0
       
        return Z
   
    def arc_length(self, G:Tensor)->Tensor:
       
        T = G.shape[0]-1
        G = G.view(T+1,-1)
        L = ((G[1:]-G[0:-1])**2).sum(1).sqrt().sum(0)
       
        return L
   
    def energy_fun(self, G:Tensor)->Tensor:
       
        T = G.shape[0]-1
        E = (((G[1:]-G[0:-1]).view(-1))**2).sum(0)
        E /= 2
        E *= T
       
        return E
   
    def jacobian_mat(self, y:Tensor, x:Tensor, create_graph:str = False)->Tensor:
       
        jac = []
        flat_y = y.reshape(-1)
        grad_y = torch.zeros_like(flat_y)
        for i in range(len(flat_y)):
            grad_y[i] = 1.0
            grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True,
                                          create_graph=create_graph)
            jac.append(grad_x.reshape(x.shape))
            grad_y[i] = 0.0
           
        return torch.stack(jac).reshape(y.shape+x.shape)
   
    def compute_geodesic(self, z_init:Tensor, epochs:int = None, lr:float = None,
                         eps:float = None, save_time:int = None,
                         save_path:str = None)->Tuple[List, Tensor]:
       
        if epochs is None:
            epochs = self.epochs
        if lr is None:
            lr = self.lr
        if eps is None:
            eps = self.eps
        if save_time is None:
            save_time = self.save_time
        if save_path is None:
            save_path = self.save_path
       
        T = len(z_init)-1
        z0 = z_init[0].view(1,-1)
        zT = z_init[-1].view(1,-1)
        z = z_init[1:T].clone().detach().requires_grad_(True)
       
        model = geodesic_path_al1(z0.to(self.device), zT.to(self.device), z.to(self.device),
                                  self.g, T, self.device).to(self.device) #Model used
        optimizer = optim.Adam(model.parameters(), lr=lr)
       
        loss = []
        E_prev = torch.tensor(0.0, device = self.device)
       
        start_time = datetime.datetime.now()
        time_diff = datetime.timedelta(hours=save_time)
       
        model.train()
        for epoch in range(epochs):
            E = model()
            #optimizer.zero_grad(set_to_none=True) #Based on performance tuning
            optimizer.zero_grad()
            E.backward()
            optimizer.step()
           
            if torch.abs(E-E_prev)<eps:
                loss.append(E.item())
                break
            else:
                E_prev = E.item()
                loss.append(E_prev)

           
           
            current_time = datetime.datetime.now()
            if current_time - start_time >= time_diff:
                print(f"Iteration {epoch+1}/{epochs} - E_fun={E.item():.4f}")
                for name, param in model.named_parameters():
                    geodesic_z = param.data
               
                geodesic_z = torch.cat((z0.view(1,-1), geodesic_z, zT.view(1,-1)), dim = 0)
                torch.save({'loss': loss,
                    'geodesic_z_new': geodesic_z},
                   save_path)
                start_time = current_time

       
        for name, param in model.named_parameters():
            geodesic_z = param.data
       
        geodesic_z = torch.cat((z0.view(1,-1), geodesic_z, zT.view(1,-1)), dim = 0)
       
        return loss, geodesic_z
   
    def compute_geodesic_fast(self, z_init:Tensor, epochs:int = None, lr:float = None,
                         eps:float = None)->Tensor:
       
        if epochs is None:
            epochs = self.epochs
        if lr is None:
            lr = self.lr
        if eps is None:
            eps = self.eps
       
        T = len(z_init)-1
        z0 = z_init[0].view(1,-1)
        zT = z_init[-1].view(1,-1)
        z = z_init[1:T].clone().detach().requires_grad_(True)
       
        model = geodesic_path_al1(z0.to(self.device), zT.to(self.device), z.to(self.device),
                                  self.g, T, self.device).to(self.device) #Model used
        optimizer = optim.Adam(model.parameters(), lr=lr)
       
        E_prev = torch.tensor(0.0, device = self.device)
        model.train()
        for epoch in range(epochs):
            E = model()
            #optimizer.zero_grad(set_to_none=True) #Based on performance tuning
            optimizer.zero_grad()
            E.backward()
            optimizer.step()
           
            if torch.abs(E-E_prev)<eps:
                break
            else:
                E_prev = E.item()

       
        for name, param in model.named_parameters():
            geodesic_z = param.data
       
        geodesic_z = torch.cat((z0.view(1,-1), geodesic_z, zT.view(1,-1)), dim = 0)
       
        return geodesic_z
   
    def Log_map(self, z0:Tensor, zT:Tensor, epochs:int=None, lr:float=None,
                eps:float = None, T:int = None)->Tuple[Tensor, Tensor, Tensor,
                                                       Tensor]:
       
        if epochs is None:
            epochs = self.epochs
        if lr is None:
            lr = self.lr
        if eps is None:
            eps = self.eps
                                                       
        z_init = self.linear_inter(z0, zT, T)
        z_geodesic = self.compute_geodesic_fast(z_init, epochs, lr, eps)
        g_geodesic = self.g(z_geodesic)
        v_g = (g_geodesic[1]-g_geodesic[0])*T
       
        shape = list(g_geodesic.shape)
        shape = [1]+shape[1:]
        x0 = g_geodesic[0].clone().detach().requires_grad_(True).view(shape)
        z0 = self.h(x0).view(-1)
        jacobi_h = self.jacobian_mat(z0, x0)
        jacobi_h = jacobi_h.view(len(z0.view(-1)), len(x0.view(-1)))
        v_z = torch.mv(jacobi_h, v_g.view(-1))
       
        return v_z, v_g, z_geodesic, g_geodesic
   
    def linear_parallel_translation(self, za:Tensor, zb:Tensor, zc:Tensor,
                                    T:int=None)->Tuple[Tensor,Tensor]:
       
        if T is None:
            T = self.T
       
        shape = [T+1]+list(za.squeeze().shape)
        v = zb-za
       
        z_linear = (torch.ones(shape,device=self.device)*v).transpose(0,1)*torch.linspace(0,1,T+1, device=self.device)
        z_linear = z_linear.transpose(0,1)+zc
           
        g_linear = self.g(z_linear)
       
        return z_linear, g_linear
   
    def parallel_translation(self, Z:Tensor, v0:Tensor)->Tuple[Tensor,Tensor]:
       
        T = Z.shape[0]-1
               
        #jacobi_g = self.get_jacobian(self.model_decoder, Z[0], n_decoder)
        z0 = Z[0].clone().detach().requires_grad_(True)
        y = self.g(z0.view(1,-1)).view(-1)
        jacobi_g = self.jacobian_mat(y, z0)
        u0 = torch.mv(jacobi_g, v0.view(-1))
       
        u_prev = u0
        for i in range(0,T):
            print(f"Iteration {i+1}/{T}")
            #xi = g[i] #is done in g for all
            zi = Z[i+1].clone().detach().requires_grad_(True)
            gi = self.g(zi.view(1,-1)).view(-1)
            jacobi_g = self.jacobian_mat(gi, zi)
            U,S,V = torch.svd(jacobi_g)
            ui = torch.mv(torch.matmul(U, torch.transpose(U, 0, 1)),u_prev)
            ui = torch.norm(u_prev)/torch.norm(ui)*ui
            u_prev = ui
       
        xT = self.g(Z[-1].view(1,-1)).detach().requires_grad_(True)
        zT = self.h(xT).view(-1)
        jacobi_h = self.jacobian_mat(zT, xT)
        jacobi_h = jacobi_h.view(len(zT.view(-1)), len(xT.view(-1)))
        vT = torch.mv(jacobi_h, ui)
        uT = ui
   
        return vT, uT
   
    def geodesic_shooting(self, z0:Tensor, u0:Tensor, T:int=None)->Tuple[Tensor,
                                                                         Tensor,
                                                                         Tensor]:
       
        if T is None:
            T = self.T
       
        delta = 1/T
        x0 = self.g(z0)
        shape = x0.shape
        zdim = [T+1]
        zdim = zdim + list((z0.squeeze()).shape)
        gdim = [T+1]
        gdim = gdim + list((x0.squeeze()).shape)
       
        Z = torch.zeros(zdim)
        G = torch.zeros(gdim)
        gdim = x0.squeeze().shape
           
        zi = z0
        xi = x0.view(-1)
        u_prev = u0.view(-1)
        Z[0] = z0
        G[0] = x0
        for i in range(0, T-1):
            print(f"Iteration {i+1}/{T}")
            xi = (xi+delta*u_prev).view(shape)
            zi = self.h(xi).view(-1)
            xi = self.g(zi.view(1,-1)).view(-1)
            jacobi_g = self.jacobian_mat(xi, zi)
            U,S,V = torch.svd(jacobi_g)
            ui = torch.mv(torch.matmul(U, torch.transpose(U, 0, 1)),u_prev.view(-1))
            ui = torch.norm(u_prev)/torch.norm(ui)*ui
            u_prev = ui
            Z[i+1] = zi
            G[i+1] = xi.view(gdim)
       
        print(f"Iteration {T}/{T}")
        xT = (xi+delta*u_prev).view(shape)
        zT = self.h(xT)
        xT = self.g(zT.view(1,-1))
        Z[-1] = zT.squeeze()
        G[-1] = xT.squeeze()
       
        return Z, G, ui
   
    def linear_distance_matrix(self, Z:Tensor, T:int = None)->Tensor:
       
        if T is None:
            T = self.T
       
        N = Z.shape[0]
        dmat = torch.zeros((N, N), device=self.device)
       
        for i in range(0, N):
            print(f"Computing row {i+1}/{N}...")
            for j in range(i+1,N):
                z_linear = self.linear_inter(Z[i], Z[j], T)
                L = self.arc_length(self.g(z_linear)).item()
               
                dmat[i][j] = L
       
        dmat += dmat.transpose(0,1)
               
        return dmat
   
    def euclidean_distance_matrix(self, X:Tensor)->Tensor:
       
        N = X.shape[0]
        dmat = torch.zeros((N, N), device=self.device)
       
        for i in range(0, N):
            print(f"Computing row {i+1}/{N}...")
            for j in range(i+1,N):
                Xdif = (X[i]-X[j]).view(-1)
                L = torch.norm(Xdif, 'fro')
               
                dmat[i][j] = L.item()
               
        dmat += dmat.transpose(0,1)
               
        return dmat
   
    def geodesic_distance_matrix(self, Z:Tensor, epochs:int = None,
                                 lr:float = None, T:int = None,
                                 eps:float = None, dmat:Tensor = None,
                                 row_idx:int = None, save_time:int = None,
                                 save_path:str = None)->Tensor:
       
        if epochs is None:
            epochs = self.epochs
        if lr is None:
            lr = self.lr
        if eps is None:
            eps = self.eps
        if T is None:
            T = self.T
        if save_time is None:
            save_time = self.save_time
        if save_path is None:
            save_path = self.save_path
           
        N = Z.shape[0]
        if dmat is None:            
            dmat = torch.zeros((N, N), device=self.device)
            row_idx = 0
       
        start_time = datetime.datetime.now()
        time_diff = datetime.timedelta(hours=save_time)
        for i in range(row_idx, N):
            print(f"Computing row {i+1}/{N}...")
            for j in range(i+1,N):
                Z_int = self.linear_inter(Z[i], Z[j], T)
                geodesic_z = self.compute_geodesic_fast(Z_int,
                                                      epochs = epochs,
                                                      lr = lr,
                                                      eps = eps)
                L = self.arc_length(self.g(geodesic_z)).item()
               
                dmat[i][j] = L
               
            current_time = datetime.datetime.now()
            if current_time - start_time >= time_diff:
                print(f"Saving row {i+1}/{N}")
                torch.save({'dmat': dmat,
                    'row_idx': i+1},
                   save_path)
                start_time = current_time
               
        dmat += dmat.transpose(0,1)
               
        return dmat
   
    def linear_mean(self, Z:Tensor)->Tuple[Tensor,Tensor]:
       
        mu_z = (torch.mean(Z, dim = 0)).view(1,-1)
        mu_g = self.g(mu_z)
       
        return mu_z, mu_g
   
    def frechet_mean(self, Z:Tensor, mu_init:Tensor, T:int=None,
                              epochs_geodesic:int = None, epochs_frechet:int = None,
                              geodesic_lr:float = None, frechet_lr:float = None,
                              eps:float = None, save_time:int = None,
                              save_path:int = None)->Tuple[List,Tensor]:
       
        if epochs_geodesic is None:
            epochs_geodesic = self.epochs
        if epochs_frechet is None:
            epochs_frechet = self.epochs
        if geodesic_lr is None:
            geodesic_lr = self.lr
        if frechet_lr is None:
            frechet_lr = self.lr
        if eps is None:
            eps = self.eps
        if T is None:
            T = self.T
        if save_time is None:
            save_time = self.save_time
        if save_path is None:
            save_path = self.save_path
       
        mu_init = mu_init.clone().detach().requires_grad_(True)
        model = frechet_mean(mu_init, self.h, self.g, T,
                             epochs=epochs_geodesic,
                             lr = geodesic_lr,
                             eps = eps,
                             device = self.device).to(self.device) #Model used

        optimizer = optim.Adam(model.parameters(), lr=frechet_lr)
       
        loss = []
        L_prev = torch.tensor(0.0, device=self.device)
       
        start_time = datetime.datetime.now()
        time_diff = datetime.timedelta(hours=save_time)
       
        model.train()
        for epoch in range(epochs_frechet):
            print(epoch)
            L = model(Z)
            #optimizer.zero_grad(set_to_none=True) #Based on performance tuning
            optimizer.zero_grad()
            L.backward()
            optimizer.step()
           
            if torch.abs(L-L_prev)<eps:
                loss.append(L.item())
                break
            else:
                L_prev = L.item()
                loss.append(L_prev)

           
            current_time = datetime.datetime.now()
            if current_time - start_time >= time_diff:
                print(f"Iteration {epoch+1}/{epochs_frechet} - L={L_prev:.4f}")
                for name, param in model.named_parameters():
                    mu_z = param.data
                   
                torch.save({'loss': loss,
                    'mu_z': mu_z},
                   save_path)
                start_time = current_time

        for name, param in model.named_parameters():
            mu = param.data
               
        return loss, mu

#%% Computes geodesics
           
class geodesic_path_al1(nn.Module):
    def __init__(self,
                 z0:Tensor,
                 zT:Tensor,
                 geodesic_z:Tensor,
                 model_decoder:Callable[[Tensor], Tensor],
                 T:int,
                 device:str
                 ):
        super(geodesic_path_al1, self).__init__()
   
        self.geodesic_z = nn.Parameter(geodesic_z, requires_grad=True)
        self.model_decoder = model_decoder
        self.g0 = (model_decoder(z0)).detach()
        self.gT = (model_decoder(zT)).detach()
        self.T = T
        self.device = device
   
    def forward(self)->Tensor:
       
        E = torch.tensor(0.0, device=self.device)
        G = self.model_decoder(self.geodesic_z)
       
        E += (((G[0]-self.g0).view(-1))**2).sum(0)
        E += (((self.gT-G[-1]).view(-1))**2).sum(0)
        E += (((G[1:]-G[0:-1]).view(-1))**2).sum(0)

        E /= 2
        E *= self.T
       
        return E

#%% Computes Fréchet mean

class frechet_mean(nn.Module):
    def __init__(self,
                 mu_init:Tensor,
                 model_encoder:Callable[[Tensor], Tensor],
                 model_decoder:Callable[[Tensor], Tensor],
                 T:int,
                 epochs:int,
                 lr:float,
                 eps:float,
                 device:str
                 ):
        super(frechet_mean, self).__init__()
           
        self.mu = nn.Parameter(mu_init, requires_grad=True)
        self.model_decoder = model_decoder
        self.T = T
        self.rm = rm_generative_models(model_encoder, model_decoder, device)
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.eps = eps
   
    def forward(self, z:Tensor)->Tensor:
       
        L = torch.tensor(0.0, device=self.device)
        N = z.shape[0]
       
        for i in range(N):
            dat = z[i]
            z_init = self.rm.linear_inter(self.mu, dat, self.T)
            g0 = self.model_decoder(self.mu.view(1,-1)).detach().view(-1)
            gT = self.model_decoder(dat.view(1,-1)).detach().view(-1)
            geodesic_z = self.rm.compute_geodesic_fast(z_init, epochs=self.epochs, lr = self.lr,
                                                       eps = self.eps)
            G = self.model_decoder(geodesic_z).detach()
           
            T = G.shape[0]-1
            G = G.view(T+1,-1)
           
            L += (((G[0]-g0).view(-1))**2).sum(0).sqrt()
            L += (((gT-G[-1]).view(-1))**2).sum(0).sqrt()
            L += ((G[1:]-G[0:-1])**2).sum(1).sqrt().sum(0)
           
            L = L**2
       
        return L