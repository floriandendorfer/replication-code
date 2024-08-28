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

# +1.24838612      0.07809104
# +1.50331878      0.08999074
# -0.00619878      0.00020315
# -10.2594999      0.10257144
# -9.72217131      0.10229196
# -9.44320539      0.10255632
# -9.07300601      0.10329358
# +2.00972201      0.11552958    

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

#[247359. 297059. 445635. 783870.   2273.   3330.   3961.   4951.]
#4097.00956 4698.90595 7776.08706 10474.4094 0.711975251 .991580160 1.65312917 2.19839718
 
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
WMax_t_l = minimize(welfare, +10, args=(theta_hat,c_hat,[cs0,ps0],[P_star,s_star,V_star],tol,params), method='Nelder-Mead')
#WMax_t_l = minimize(welfare, [7,8,9,10], args=(theta_hat,c_hat,[cs0,ps0],[P_star,s_star,V_star],tol,params), method='Nelder-Mead')

    # Current version
#Entrant tax/subsidy is [82.8790625]
#Incumbent tax/subsidy is [-13.90044991]
#Consumer surplus change is [[214470.10647843]]
#Producer surplus change is [-23624.91984898]
#Welfare change is [[954.22593315]]
#Change in # properties: 32

    # "Freezing" revenue
#Entrant tax/subsidy is [71.764125]
#Incumbent tax/subsidy is [-54.85708589]
#Consumer surplus change is [[76196.94533622]]
#Producer surplus change is [-30125.23055981]
#Welfare change is [[46071.71477642]]
#Change in # properties: 24

    # Revenue-neutrak (avg) entrant subsidy
#Entrant tax/subsidy is [6.31634521]
#Incumbent tax/subsidy is [0.]
#Consumer surplus change is [[3342.30873484]]
#Producer surplus change is [-2599.73724795]
#Welfare change is [[742.57148689]]
#Change in # properties: -7

    # Incumbent tax with lump-sum reimbursement to incumbents
#Incumbent tax/subsidy is [-22.19474792]
#Consumer surplus change is [[178372.30560608]]
#Producer surplus change is [11668.01383812]
#Welfare change is [[190040.3194442]]
#Change in # properties: 212

t = np.ones((S.shape[0],1))*0 
t[210:231,:] = np.array([-54.85712764]).T
t[441:462,:] = np.array([-54.85712764]).T
t[672:693,:] = np.array([-54.85712764]).T
t[903:924,:] = np.array([-54.85712764]).T
t[[0,231,462,693],:] = np.array([71.7641875]).T

V_c,s_c,P_c,chi_c,lamb_c = solver(theta_hat,c_hat,[P_star,s_star,V_star],t,tol,params)

q_c = q_s(P_c,P_c,s_c,theta_hat,t,params)    
T_c = T_s(q_c,theta_hat,params)
lamb0_c = np.array([np.repeat(lamb_c,231)]).T
chi0_c = np.array([chi_c]).T
eV0_in = T_c @ V_c
eV0_out = V_c.reshape((231,4),order='F')[0,:]
psc1_out = -(J/4-s_c[0,:231].sum())*(c_hat[0] - (1-lamb0_c[0])*(delta*eV0_out[0] + c_hat[0]))
psc2_out = -(J/4-s_c[0,231:462].sum())*(c_hat[1] - (1-lamb0_c[231])*(delta*eV0_out[1] + c_hat[1]))
psc3_out = -(J/4-s_c[0,462:693].sum())*(c_hat[2] - (1-lamb0_c[462])*(delta*eV0_out[2] + c_hat[2]))
psc4_out = -(J/4-s_c[0,693:].sum())*(c_hat[3] - (1-lamb0_c[693])*(delta*eV0_out[3] + c_hat[3]))
psc_in = (s_c.T *( 28*q_star*(1+f)*P_star.T - (np.array([np.repeat(c_hat[4:],231)]).T - chi0_c * (delta*eV0_in + np.array([np.repeat(c_hat[4:],231)]).T)) ) ).sum()
gs_c = (s_c.T *( - 28*q_star*P_star.T + 28*q_c*(P_c.T-t) )).sum()
cs0 = -( mu - s_c @ ( mu*ccp_s(P_c,P_c,s_c,theta_hat,0,params) - q_c ) )*28*np.log(1 + (s_c @ np.array([np.diagonal(np.exp(U(P_c,theta_hat,0,params)))]).T) )/theta_hat[2]
psc = psc_in + psc1_out + psc2_out + psc3_out + psc4_out

    # TOTAL
(s_c - s_star).sum()
100*(s_c - s_star).sum()/s_star.sum()
(s_c @ (P_c.T-t))/s_c.sum() - (s_star @ P_star.T)/s_star.sum()
100*((s_c @ (P_c.T-t))/s_c.sum() - (s_star @ P_star.T)/s_star.sum())/((s_star @ P_star.T)/s_star.sum())
(s_c @ P_c.T)/s_c.sum() - (s_star @ P_star.T)/s_star.sum()
100*((s_c @ P_c.T)/s_c.sum() - (s_star @ P_star.T)/s_star.sum())/((s_star @ P_star.T)/s_star.sum())
(s_c @ q_c)/s_c.sum() - (s_star @ q_star)/s_star.sum()
100*((s_c @ q_c)/s_c.sum() - (s_star @ q_star)/s_star.sum())/((s_star @ q_star)/s_star.sum())

    # ENTRANT
100*t[0]/((P_star[:,[0,231,462,693]] @ s_star[:,[0,231,462,693]].T)/s_star[:,[0,231,462,693]].sum())
(s_c[:,[0,231,462,693]] - s_star[:,[0,231,462,693]]).sum()
100*((s_c[:,[0,231,462,693]] - s_star[:,[0,231,462,693]]).sum()/s_star[:,[0,231,462,693]].sum())
(s_c[:,[0,231,462,693]] @ P_c[:,[0,231,462,693]].T)/s_c[:,[0,231,462,693]].sum() - (s_star[:,[0,231,462,693]] @ P_star[:,[0,231,462,693]].T)/s_star[:,[0,231,462,693]].sum()
100*((s_c[:,[0,231,462,693]] @ P_c[:,[0,231,462,693]].T)/s_c[:,[0,231,462,693]].sum() - (s_star[:,[0,231,462,693]] @ P_star[:,[0,231,462,693]].T)/s_star[:,[0,231,462,693]].sum())/((s_star[:,[0,231,462,693]] @ P_star[:,[0,231,462,693]].T)/s_star[:,[0,231,462,693]].sum())
(s_c[:,[0,231,462,693]] @ (P_c[:,[0,231,462,693]].T-t[0]))/s_c[:,[0,231,462,693]].sum() - (s_star[:,[0,231,462,693]] @ P_star[:,[0,231,462,693]].T)/s_star[:,[0,231,462,693]].sum()
100*((s_c[:,[0,231,462,693]] @ (P_c[:,[0,231,462,693]].T-t[0]))/s_c[:,[0,231,462,693]].sum() - (s_star[:,[0,231,462,693]] @ P_star[:,[0,231,462,693]].T)/s_star[:,[0,231,462,693]].sum())/((s_star[:,[0,231,462,693]] @ P_star[:,[0,231,462,693]].T)/s_star[:,[0,231,462,693]].sum())
(s_c[:,[0,231,462,693]] @ q_c[[0,231,462,693],:])/s_c[:,[0,231,462,693]].sum() - (s_star[:,[0,231,462,693]] @ q_star[[0,231,462,693],:])/s_star[:,[0,231,462,693]].sum()
100*((s_c[:,[0,231,462,693]] @ q_c[[0,231,462,693],:])/s_c[:,[0,231,462,693]].sum() - (s_star[:,[0,231,462,693]] @ q_star[[0,231,462,693],:])/s_star[:,[0,231,462,693]].sum())/((s_star[:,[0,231,462,693]] @ q_star[[0,231,462,693],:])/s_star[:,[0,231,462,693]].sum())

    # INCUMBENT
n_inc = (s_c[:,210:231] + s_c[:,441:462] + s_c[:,672:693] + s_c[:,903:924]).sum()
n_inc + ( - s_star[:,210:231] - s_star[:,441:462] - s_star[:,672:693] - s_star[:,903:924]).sum()
P_inc = (P_c[:,210:231] @ s_c[:,210:231].T + P_c[:,441:462] @ s_c[:,441:462].T + P_c[:,672:693] @ s_c[:,672:693].T + P_c[:,903:924] @ s_c[:,903:924].T).sum()/n_inc
100*t[-1]/P_inc
100*(n_inc + (- s_star[:,210:231] - s_star[:,441:462] - s_star[:,672:693] - s_star[:,903:924]).sum())/(s_star[:,210:231] + s_star[:,441:462] + s_star[:,672:693] + s_star[:,903:924]).sum()
P_inc - (s_star[:,210:231] @ P_star[:,210:231].T  + s_star[:,441:462] @ P_star[:,441:462].T + s_star[:,672:693] @ P_star[:,672:693].T + s_star[:,903:924] @ P_star[:,903:924].T)/(s_star[:,210:231] + s_star[:,441:462] + s_star[:,672:693] + s_star[:,903:924]).sum() 
100*(P_inc - (s_star[:,210:231] @ P_star[:,210:231].T  + s_star[:,441:462] @ P_star[:,441:462].T + s_star[:,672:693] @ P_star[:,672:693].T + s_star[:,903:924] @ P_star[:,903:924].T)/(s_star[:,210:231] + s_star[:,441:462] + s_star[:,672:693] + s_star[:,903:924]).sum() 
)/((s_star[:,210:231] @ P_star[:,210:231].T  + s_star[:,441:462] @ P_star[:,441:462].T + s_star[:,672:693] @ P_star[:,672:693].T + s_star[:,903:924] @ P_star[:,903:924].T)/(s_star[:,210:231] + s_star[:,441:462] + s_star[:,672:693] + s_star[:,903:924]).sum())
(s_c[:,210:231] @ (P_c[:,210:231].T-t[-1])  + s_c[:,441:462] @ (P_c[:,441:462].T-t[-1]) + s_c[:,672:693] @ (P_c[:,672:693].T-t[-1]) + s_c[:,903:924] @ (P_c[:,903:924].T-t[-1]))/n_inc  - (s_star[:,210:231] @ P_star[:,210:231].T  + s_star[:,441:462] @ P_star[:,441:462].T + s_star[:,672:693] @ P_star[:,672:693].T + s_star[:,903:924] @ P_star[:,903:924].T)/(s_star[:,210:231] + s_star[:,441:462] + s_star[:,672:693] + s_star[:,903:924]).sum() 
100*((s_c[:,210:231] @ (P_c[:,210:231].T-t[-1])  + s_c[:,441:462] @ (P_c[:,441:462].T-t[-1]) + s_c[:,672:693] @ (P_c[:,672:693].T-t[-1]) + s_c[:,903:924] @ (P_c[:,903:924].T-t[-1]))/n_inc - (s_star[:,210:231] @ P_star[:,210:231].T  + s_star[:,441:462] @ P_star[:,441:462].T + s_star[:,672:693] @ P_star[:,672:693].T + s_star[:,903:924] @ P_star[:,903:924].T)/(s_star[:,210:231] + s_star[:,441:462] + s_star[:,672:693] + s_star[:,903:924]).sum() 
)/((s_star[:,210:231] @ P_star[:,210:231].T  + s_star[:,441:462] @ P_star[:,441:462].T + s_star[:,672:693] @ P_star[:,672:693].T + s_star[:,903:924] @ P_star[:,903:924].T)/(s_star[:,210:231] + s_star[:,441:462] + s_star[:,672:693] + s_star[:,903:924]).sum())
(s_c[:,210:231] @ q_c[210:231,:]  + s_c[:,441:462] @ q_c[441:462,:] + s_c[:,672:693] @ q_c[672:693,:] + s_c[:,903:924] @ q_c[903:924,:])/n_inc - (s_star[:,210:231] @ q_star[210:231,:]  + s_star[:,441:462] @ q_star[441:462,:] + s_star[:,672:693] @ q_star[672:693,:] + s_star[:,903:924] @ q_star[903:924,:])/(s_star[:,210:231] + s_star[:,441:462] + s_star[:,672:693] + s_star[:,903:924]).sum() 
100*((s_c[:,210:231] @ q_c[210:231,:]  + s_c[:,441:462] @ q_c[441:462,:] + s_c[:,672:693] @ q_c[672:693,:] + s_c[:,903:924] @ q_c[903:924,:])/n_inc - (s_star[:,210:231] @ q_star[210:231,:]  + s_star[:,441:462] @ q_star[441:462,:] + s_star[:,672:693] @ q_star[672:693,:] + s_star[:,903:924] @ q_star[903:924,:])/(s_star[:,210:231] + s_star[:,441:462] + s_star[:,672:693] + s_star[:,903:924]).sum() )/((s_star[:,210:231] @ q_star[210:231,:]  + s_star[:,441:462] @ q_star[441:462,:] + s_star[:,672:693] @ q_star[672:693,:] + s_star[:,903:924] @ q_star[903:924,:])/(s_star[:,210:231] + s_star[:,441:462] + s_star[:,672:693] + s_star[:,903:924]).sum() )