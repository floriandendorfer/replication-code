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

os.chdir(os.path.dirname(os.path.abspath(__file__)))

################################# Parameters ##################################

delta = .995
tol = 1e-6
f = .142
upsilon_r = .992
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

################################ Loading Data #################################

data = pd.read_pickle(os.path.join(Path().absolute().parent, 'data\\data.pkl'))

############################## Demand Estimation ##############################

    # inversion
data_est = data.copy()
data_est['share'] = -np.log(1 - data_est['q'])/mu
data_est = data_est[(data_est['share'] > 0) & (data_est['share'] < np.inf)]
data_est['share_0'] = 1 - data_est.groupby(['month'])['share'].transform('sum')
data_est['r'] = 1 + 4*data_est['K']/data_est['N']
data_est = data_est[(data_est['r'] > 0)]

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
S1_hat = (np.diag(np.linalg.inv((G_bar.T @ W2) @ G_bar))/len(data_est))**.5

    # estimation results
print('Estimates:',res_demand.x)
print('Standard errors:',S1_hat)

###############################################################################
############################## Supply Estimation ##############################
###############################################################################

from functions import likelihood
from functions import q_s

    # initial guess
P_init = np.array([[200]*len(S)])
s_init = np.zeros((S.shape[0],1)).T
s_init[0,[0,231,462,693]] = [J/8,J/8,J/8,J/8]
V_init = (28*q_s(200,P_init,s_init,theta_hat,0,params)*P_init.T)/(1-delta)

    # empirical state distribution
s_d = np.array([(data.groupby(['x']).size()/52).reindex(np.arange(0,len(S)), fill_value=0)])

    # Store initital guess
shelve_file = shelve.open("guess") 
shelve_file['guess'] = [P_init,s_init,V_init] 
shelve_file.close() 

    # maximization
tol=1e-0
k0 = np.log([100000,100000,100000,100000,3000,3000,3000,3000])
res_supply = minimize(likelihood, k0, args=(theta_hat,tol,s_d,params), method='Nelder-Mead')
c_hat = np.exp(res_supply.x)

    # standard errors
from functions import approx_score
scores_x = approx_score(res_supply.x, 1e-10,theta_hat,[P_init,s_init,V_init],tol,params)
scores = scores_x[:,np.hstack((np.array(data['x']))).astype(int)]
S2_hat = ((c_hat * np.diag(np.linalg.inv(scores@scores.T)))* c_hat)**0.5

    # estimation results
print('Estimates:',c_hat)
print('Standard errors:',S2_hat)

###############################################################################
############################ Counterfactual Analysis ##########################
###############################################################################

    # model solution
from functions import solver,U,ccp_s,T_s
V_star,s_star,P_star,chi_star,lamb_star = solver(theta_hat,c_hat,[P_init,s_init,V_init],0,tol,params)

q_star = q_s(P_star,P_star,s_star,theta_hat,0,params)    
T_star = T_s(q_star,theta_hat,params)
lamb0_star = np.array([np.repeat(lamb_star,231)]).T
chi0_star = np.array([chi_star]).T
eV0_in = T_star @ V_star
eV0_out = V_star.reshape((231,4),order='F')[0,:]
ps01_out = -(J/4-s_star[0,:231].sum())*(c_hat[0] - (1-lamb0_star[0])*(delta*eV0_out[0] + c_hat[0]))
ps02_out = -(J/4-s_star[0,231:462].sum())*(c_hat[1] - (1-lamb0_star[231])*(delta*eV0_out[1] + c_hat[1]))
ps03_out = -(J/4-s_star[0,462:693].sum())*(c_hat[2] - (1-lamb0_star[462])*(delta*eV0_out[2] + c_hat[2]))
ps04_out = -(J/4-s_star[0,693:].sum())*(c_hat[3] - (1-lamb0_star[693])*(delta*eV0_out[3] + c_hat[3]))
ps0_in = (s_star.T *( 28*q_star*(1+f)*P_star.T - (np.array([np.repeat(c_hat[4:],231)]).T - chi0_star * (delta*eV0_in + np.array([np.repeat(c_hat[4:],231)]).T)) ) ).sum()
cs0 = -( mu - s_star @ ( mu*ccp_s(P_star,P_star,s_star,theta_hat,0,params) - q_star ) )*28*np.log(1 + (s_star @ np.array([np.diagonal(np.exp(U(P_star,theta_hat,0,params)))]).T) )/theta_hat[2]
ps0 = ps0_in + ps01_out + ps02_out + ps03_out + ps04_out

from functions import welfare
WMax_t_l = minimize(welfare, 0, args=(theta_hat,c_hat,[cs0,ps0],[P_star,s_star,V_star],tol,params), method='Nelder-Mead')
