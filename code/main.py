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
import shelve
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath('__file__')))

###### Parameters ######

    # demand
a = 12.2260
b = 2.1134
alpha = -0.0068
beta = np.array([-12.5906,-12.1095,-11.7011,-11.3012])
gamma = 4.8860
theta = [a,b,alpha,beta,gamma]

    # supply
kappa_bar = [55496,96673,161946,270623]
phi_bar = [2580,3577,4562,5751]
c = np.append(kappa_bar,phi_bar)

    # other
delta = .995
tol = 1e-6
f = .142
upsilon_r = 0.7041
nmax = 20
mu = 10000
J = 10000

    # type space
from functions import cartesian_product
S = cartesian_product(np.arange(nmax+1),np.arange(nmax+1))
S = S[S[:,0]<=S[:,1]]
S=S[np.lexsort((S[:,0],S[:,1]))]
S=np.vstack((np.hstack((S,np.repeat(np.array([[1,0,0,0]]),len(S),axis=0))),
             np.hstack((S,np.repeat(np.array([[0,1,0,0]]),len(S),axis=0))),
             np.hstack((S,np.repeat(np.array([[0,0,1,0]]),len(S),axis=0))),
             np.hstack((S,np.repeat(np.array([[0,0,0,1]]),len(S),axis=0)))
             ))   
params = [delta,f,J,mu,nmax,S,upsilon_r]

###### Solving The Model ######

from functions import q_s,solver

P_init = np.array([[300]*len(S)])
s_init = np.array([[J/(2*len(S))]*len(S)])
V_init = (30*q_s(300,P_init,s_init,theta,0,params)*P_init.T)/(1-delta)

V_star,s_star,P_star,chi_star,lamb_star = solver(theta,c,[P_init,s_init,V_init],tol,params)

###### Data Generating Process ######

for t in range(1,13*4+1):
    index = np.repeat(range(0,924),np.random.multinomial(s_star.sum().astype(int),(s_star/s_star.sum()).flatten(),size=(13*4,))[t-1,:])
    p = (P_star.T + np.random.normal(loc = 0, scale = 25.0, size = (P_star.T).shape))
    if t == 1:
        data = np.hstack(( np.zeros((len(index),1)) + t, np.array([index]).T, S[index,:], p[index,:], q_s(p,p,s_star,theta,0,params)[index,:] + np.random.normal(loc = 0, scale = 0.1, size = q_s(p,P_star,s_star,theta,0,params)[index,:].shape ) ))
    else:
        data = np.vstack((data, np.hstack(( np.zeros((len(index),1)) + t, np.array([index]).T, S[index,:], p[index,:], q_s(p,p,s_star,theta,0,params)[index,:] + np.random.normal(loc = 0, scale = 0.1, size = q_s(P_star,P_star,s_star,theta,0,params)[index,:].shape ) )) ))

data = pd.DataFrame(data,columns=['period','x','K','N','type 1','type 2','type 3','type 4','p','q'])

data.to_pickle(os.path.join(Path().absolute().parent, 'data\\data.pkl'))
data = pd.read_pickle(os.path.join(Path().absolute().parent, 'data\\data.pkl'))

###### Demand Estimation ######

    # inversion
data_est = data.copy()
data_est['share'] = -np.log(1 - data_est['q'])/mu
data_est = data_est[data_est['share'] > 0]
data_est['share_0'] = 1 - data_est.groupby(['period'])['share'].transform('sum')
data_est['r'] = 1 + 4*data_est['K']/data_est['N']
data_est = data_est[(data_est['share'] > 0) & (data_est['r'] >0)]

    # minimization

from functions import dO,dU,O,xi,Z
omicron0 = [0,0,0,-10,-10,-10,-10,0]

W1 = np.linalg.inv( ((Z(omicron0,data_est,params)).T @ (Z(omicron0,data_est,params)))/len(data_est))
res_demand = minimize(O, omicron0, args=(data_est,W1,params), method='BFGS',jac=dO)
xi_hat = xi(res_demand.x,data_est,params)

W2 = np.linalg.inv( ((xi_hat*Z(res_demand.x,data_est,params)).T @ (xi_hat*Z(res_demand.x,data_est,params)))/len(data_est) )
res_demand = minimize(O, omicron0, args=(data_est,W2,params), method='BFGS',jac=dO)

theta_hat = [expit(res_demand.x[0])*np.exp(res_demand.x[1]),
(1-expit(res_demand.x[0]))*np.exp(res_demand.x[1]),
res_demand.x[2],
res_demand.x[3:7],
res_demand.x[7]]

    # standard errors
G_bar = ( (Z(res_demand.x,data_est,params).T @ (-dU(res_demand.x,data_est,params))) )/len(data_est)
W2 = np.linalg.inv( ((xi_hat*Z(res_demand.x,data_est,params)).T @ (xi_hat*Z(res_demand.x,data_est,params)))/len(data_est) )
S1_hat = np.diag(np.linalg.inv((G_bar.T @ W2) @ G_bar))**.5/len(data_est)

    # estimation results
print('Estimates:',res_demand.x)
print('Standard errors:',S1_hat)

###### Supply Estimation ######

from functions import likelihood

s_d = np.array([(data.groupby(['x'])['period'].count()/52).reindex(np.arange(0,len(S)), fill_value=0)])
shelve_file = shelve.open("guess") 
shelve_file['guess'] = [P_init,s_init,V_init] 
shelve_file.close() 

    # maximization
tol=1e-0
k0 = np.log([100000,100000,100000,100000,3000,3000,3000,3000])
res_supply = minimize(likelihood, k0, args=(theta_hat,tol,s_d,params), method='BFGS')
c_hat = np.exp(res_supply.x)
H_inv = res_supply.hess_inv   

    # standard errors
S2_hat = ((c_hat * np.diag(H_inv) * c_hat)/len(data))**0.5

    # estimation results
print('Estimates:',c_hat)
print('Standard errors:',S2_hat)

###### Counterfactual Analysis ######

from functions import simulation1,simulation2,Sub_prim,t_prim

    # counterfactual 1    
Wmax_Sub = minimize(Sub_prim, [0,0,0,0], args=(theta_hat,c_hat,[P_star,s_star,V_star],130,params), method='BFGS')
Sub_c = np.zeros((S.shape[0],1))
Sub_c[[0,231,462,693],:] = np.array([Wmax_Sub.x]).T

    # counterfactual 2
W1_c,CS1_c,PS1_c,GS1_c,P1_c,s1_c,V1_c = simulation1(theta_hat,c_hat,[P_star,s_star,V_star],np.zeros((S.shape[0],1)),Sub_c,1000,params)
W1_c,CS1_c,PS1_c,GS1_c,P1_c,s1_c,V1_c = simulation1(theta_hat,c_hat,[P1_c,s1_c,V1_c],np.zeros((S.shape[0],1)),Sub_c,130,params)
WMax_t = minimize(t_prim, [0,0,0,0], args=(theta_hat,c_hat,[P1_c,s1_c,V1_c],[409.51, 646.95, 945.56, 1309.57],130,params), method='BFGS')
t_c = np.zeros((S.shape[0],1))
t_c[[0,231,462,693],:] = np.array([WMax_t.x]).T
W2_c,CS2_c,PS2_c,GS2_c,P2_c,s2_c,V2_c = simulation2(theta_hat,c_hat,[P1_c,s1_c,V1_c],t_c,Sub_c,130,params)
