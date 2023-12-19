# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 09:20:40 2023

@author: Florian Dendorfer
"""

import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize
from scipy.special import expit

os.chdir(os.path.dirname(os.path.abspath('__file__')))

# Define parameters
a = 12.2260
b = 2.1134
alpha = -0.0068
beta = np.array([-12.5906,-12.1095,-11.7011,-11.3012])
gamma = 4.8860

theta = [a,b,alpha,beta,gamma]

kappa_bar = [55496,96673,161946,270623]
phi_bar = [2580,3577,4562,5751]

c = np.append(kappa_bar,phi_bar)

# Define type space

from functions import cartesian_product

delta = .995
tol = 1e-6
f = .142
upsilon_r = 0.7041
nmax = 20
mu = 10000
J = 10000

    # ...
S = cartesian_product(np.arange(nmax+1),np.arange(nmax+1))

    # ...
S = S[S[:,0]<=S[:,1]]

    # ...
S=S[np.lexsort((S[:,0],S[:,1]))]

    # ...
S=np.vstack((np.hstack((S,np.repeat(np.array([[1,0,0,0]]),len(S),axis=0))),
             np.hstack((S,np.repeat(np.array([[0,1,0,0]]),len(S),axis=0))),
             np.hstack((S,np.repeat(np.array([[0,0,1,0]]),len(S),axis=0))),
             np.hstack((S,np.repeat(np.array([[0,0,0,1]]),len(S),axis=0)))
             ))

params = [delta,f,J,mu,nmax,S,upsilon_r]

# Load functions

    # Load all functions
from functions import q_s,solver

P_init = np.array([[300]*len(S)])
s_init = np.array([[J/(2*len(S))]*len(S)])
V_init = (30*q_s(300,P_init,s_init,theta,0,params)*P_init.T)/(1-delta)

V_star,s_star,P_star,chi_star,lamb_star = solver(theta,c,[P_init,s_init,V_init],tol,params)
s_star = np.where(s_star<0,0,s_star)

# Generating 4 years of data
for t in range(1,13*4+1):
    if t == 1:
        index = np.repeat(range(0,924),np.random.multinomial(s_star.sum().astype(int),(s_star/s_star.sum()).flatten(),size=(13*4,))[t-1,:])
        p = (P_star.T + np.random.normal(loc = 0, scale = 25.0, size = (P_star.T).shape))
        data = np.hstack(( np.zeros((len(index),1)) + t, np.array([index]).T, S[index,:], p[index,:], q_s(p,p,s_star,theta,0,params)[index,:] + np.random.normal(loc = 0, scale = 0.1, size = q_s(p,P_star,s_star,theta,0,params)[index,:].shape ) ))
    else:
        p = (P_star.T + np.random.normal(loc = 0, scale = 25.0, size = (P_star.T).shape))
        data = np.vstack((data, np.hstack(( np.zeros((len(index),1)) + t, np.array([index]).T, S[index,:], p[index,:], q_s(p,p,s_star,theta,0,params)[index,:] + np.random.normal(loc = 0, scale = 0.1, size = q_s(P_star,P_star,s_star,theta,0,params)[index,:].shape ) )) ))

data = np.where(data<0,0,data)

data = pd.DataFrame(data,columns=['period','x','K','N','type 1','type 2','type 3','type 4','p','q'])

data.to_pickle('data/data.pkl')


# Inversion

    # Compute market shares
data['share'] = -np.log(1 - data['q'])/mu
data = data[data['share'] > 0]
data['share_0'] = 1 - data.groupby(['period'])['share'].transform('sum')
data['r'] = 1 + 4*data['K']/data['N']
data = data[(data['share'] > 0) & (data['r'] >0)]

# GMM

start_values = [0,0,0,-10,-10,-10,-10,0]

from functions import dO,dU,O,xi,Z

    # 1st stage
W1 = np.linalg.inv( ((Z(start_values,data,params)).T @ (Z(start_values,data,params)))/len(data))
#W1 = np.identity(8)
res_demand = minimize(O, start_values, args=(data,W1,params), method='BFGS',jac=dO)
xi_hat = xi(res_demand.x,data,params)

    # 2nd stage
W2 = np.linalg.inv( ((xi_hat*Z(res_demand.x,data,params)).T @ (xi_hat*Z(res_demand.x,data,params)))/len(data) )
res_demand = minimize(O, start_values, args=(data,W2,params), method='BFGS',jac=dO)

theta_hat = [expit(res_demand.x[0])*np.exp(res_demand.x[1]),
(1-expit(res_demand.x[0]))*np.exp(res_demand.x[1]),
res_demand.x[2],
res_demand.x[3:7],
res_demand.x[7]]

    # Standard errors
G_bar = ( (Z(res_demand.x,data,params).T @ (-dU(res_demand.x,data,params))) )/len(data)
W2 = np.linalg.inv( ((xi_hat*Z(res_demand.x,data,params)).T @ (xi_hat*Z(res_demand.x,data,params)))/len(data) )
S_hat = np.diag(np.linalg.inv((G_bar.T @ W2) @ G_bar))**.5/len(data)
print('Heteroskedasticity:',S_hat)

# Supply estimation

from functions import l

    # Compute empirical state distribution
s_d = np.array([(data.groupby(['x'])['period'].count()/data.groupby(['period']).mean().shape[0]).reindex(np.arange(0,len(S)), fill_value=0)])

    # Set tolerance level
tol = 1e-3

k0 = np.log([50000,100000,150000,200000,2500,3500,4500,5500])
 
    # Maximize log-likelihood.
res_supply = minimize(l, k0, args=(theta_hat,[P_init,s_init,V_init],tol,s_d,params), method='BFGS')

    # Standard errors.
