# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 09:20:42 2023

@author: Florian Dendorfer
"""

import numpy as np
from scipy.special import expit

#################################  Functions  #################################

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def U(p,theta,t,S,f):
    a,b,alpha,beta,gamma=theta
    k = S[:,0:1,:]
    n = S[:,1:2,:]
    return gamma*( (a+k)/(a+b+n) ) + np.repeat(np.array([beta])[np.newaxis,:, :],len(S),axis=0) + alpha*(p*f-t)

def q(p,P,s,theta,t,S,f,mu):
    a,b,alpha,beta,gamma = theta
    U_j = U(p,theta,t,S,f)
    #U_all = np.diagonal(U(P,theta,t,S,f),axis1=0,axis2=1).reshape((len(S)*4,1))
    U_all = U(P,theta,t,S,f).reshape((len(S)*4,1))
    q = 1 - np.exp(-mu*(np.exp(U_j)/(1 + s.reshape((1,len(S)*4)) @ np.exp(U_all) )))
    return q

def pi(P,s,theta,t,S,f,mu,prices):
    return q(np.repeat(np.repeat(prices,len(S),axis=0)[:,:,np.newaxis],4,axis=2),P,s,theta,t,S,f,mu)*np.repeat(np.repeat(prices,len(S),axis=0)[:,:,np.newaxis],4,axis=2)