# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 09:20:42 2023

@author: Florian Dendorfer
"""

import numpy as np
from scipy.sparse import csr_matrix

#################################  Functions  #################################

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def U_s(p,theta,t,params):
    a,b,alpha,beta,gamma,sigma=theta
    delta,f,J,mu,nmax,S,upsilon_r = params
    return gamma*( (a+S[:,0:1])/(a+b+S[:,1:2]) ) + S[:,2:] @ beta + alpha*(p*(1+f) - t)

def ccp_jg_s(p,P,s,theta,t,params):
    a,b,alpha,beta,gamma,sigma = theta
    delta,f,J,mu,nmax,S,upsilon_r = params
    U_j = np.array([np.diagonal(U_s(p,theta,t,S,f),axis1=0,axis2=1)]).T
    U_all = np.array([np.diagonal(U_s(p,theta,t,S,f),axis1=0,axis2=1)]).T
    
    return np.exp(U_j/(1-sigma))/( np.exp(U_j/(1-sigma)) - np.tile(np.exp(U_all/(1-sigma)),(1,U_j.shape[1]))
                                  + S[:,2:] @  np.array([[ float(s[:,:231] @ np.exp(U_all/(1-sigma))[:231,:]),
                                                          float(s[:,231:462] @ np.exp(U_all/(1-sigma))[231:462,:]),
                                                          float(s[:,462:693] @ np.exp(U_all/(1-sigma))[462:693,:]),
                                                          float(s[:,693:] @ np.exp(U_all/(1-sigma))[693:,:]) 
                                                          ]]).T)

def ccp_g_s(p,P,s,theta,t,params):
    a,b,alpha,beta,gamma,sigma = theta
    delta,f,J,mu,nmax,S,upsilon_r = params
    U_j = np.array([np.diagonal(U_s(p,theta,t,params),axis1=0,axis2=1)]).T
    U_all = np.array([np.diagonal(U_s(P,theta,t,params),axis1=0,axis2=1)]).T
    
    D = ( np.exp(U_j/(1-sigma)) - np.tile(np.exp(U_all/(1-sigma)),(1,U_j.shape[1]))
                                  + S[:,2:] @  np.array([[ float(s[:,:231] @ np.exp(U_all/(1-sigma))[:231,:]),
                                                          float(s[:,231:462] @ np.exp(U_all/(1-sigma))[231:462,:]),
                                                          float(s[:,462:693] @ np.exp(U_all/(1-sigma))[462:693,:]),
                                                          float(s[:,693:] @ np.exp(U_all/(1-sigma))[693:,:]) 
                                                          ]]).T)
    return D**(1-sigma)/( 1 + ((D[[0,231,462,693],0])**(1-sigma)).sum() )

def ccp_s(p,P,s,theta,t,params):
    a,b,alpha,beta,gamma,sigma = theta
    delta,f,J,mu,nmax,S,upsilon_r = params
    return ccp_jg_s(p,P,s,theta,t,params)*ccp_g_s(p,P,s,theta,t,params) 

def q_s(p,P,s,theta,t,S,f,mu):
    a,b,alpha,beta,gamma = theta
    U_j = U_s(p,theta,t,S,f)
    U_all = U_s(P,theta,t,S,f).reshape((len(S)*4,1))
    q = 1 - np.exp(-mu*(np.exp(U_j)/(1 + s.reshape((1,len(S)*4)) @ np.exp(U_all) )))
    return q

def dccp_s(p,P,s,theta,t,params):
    a,b,alpha,beta,gamma,sigma = theta
    delta,f,J,mu,nmax,S,upsilon_r = params
    ccp = ccp_s(p,P,s,theta,t,params)
    ccp_jg = ccp_jg_s(p,P,s,theta,t,params)
    return ccp*(1-sigma*ccp_jg - (1-sigma)*ccp)*alpha*(1+f)/(1-sigma)

def d2ccp_s(p,P,s,theta,t,params):
    a,b,alpha,beta,gamma,sigma = theta
    delta,f,J,mu,nmax,S,upsilon_r = params
    ccp = ccp_s(p,P,s,theta,t,params)
    ccp_jg = ccp_jg_s(p,P,s,theta,t,params)
    return ((1/(1-sigma)-(sigma/(1-sigma))*ccp_jg - 2*ccp)*(1/(1-sigma)-(sigma/(1-sigma))*ccp_jg - ccp)*ccp 
            - (sigma/((1-sigma)**2))*ccp*ccp_jg*(1-ccp_jg)
            )*alpha*(1+f)*alpha*(1+f)

def dq_s(p,P,s,theta,t,params):
    a,b,alpha,beta,gamma,sigma = theta
    delta,f,J,mu,nmax,S,upsilon_r = params
    return mu*np.exp(-mu*ccp_s(p,P,s,theta,t,params))*dccp_s(p,P,s,theta,t,params)

def d2q_s(p,P,s,theta,t,params):
    a,b,alpha,beta,gamma,sigma = theta
    delta,f,J,mu,nmax,S,upsilon_r = params
    ccp = ccp_s(p,P,s,theta,t,params)
    dccp = dccp_s(p,P,s,theta,t,params)
    d2ccp = d2ccp_s(p,P,s,theta,t,params)
    return mu*np.exp(-mu*ccp)*( d2ccp - mu*dccp**2 )

def V_s(p,P,s,V,theta,phi_bar,t,Sub,params):
    a,b,alpha,beta,gamma,sigma = theta
    delta,f,J,mu,nmax,S,upsilon_r = params
    q =q_s(p,P,s,theta,t,params)
    T = T_s(p,P,s,q,theta,t,params)
    eV = T @ V
    V = 30*q*p.T + Sub + delta*eV + np.exp(-delta*eV/phi_bar)*phi_bar
    return V
    
def dV_s(p,P,s,V,theta,phi_bar,t,params):
    a,b,alpha,beta,gamma,sigma = theta
    delta,f,J,mu,nmax,S,upsilon_r = params
    q = q_s(p,P,s,theta,t,params)
    dq = dq_s(p,P,s,theta,t,params)
    dT = dT_s(p,P,s,dq,theta,t,params)
    T = T_s(p,P,s,q,theta,t,params)
    deV = dT @ V
    dV = 30*(q + dq*p.T) + delta*deV - np.exp(-delta*(T @ V)/phi_bar)*delta*deV
    return dV.T

def d2V_s(p,P,s,V,theta,phi_bar,t,params):
    a,b,alpha,beta,gamma,sigma = theta
    delta,f,J,mu,nmax,S,upsilon_r = params
    q = q_s(p,P,s,theta,t,params)
    dq = dq_s(p,P,s,theta,t,params)
    d2q = d2q_s(p,P,s,theta,t,params)
    T = T_s(p,P,s,q,theta,t,params)
    dT = dT_s(p,P,s,dq,theta,t,params)
    d2T = d2T_s(p,P,s,d2q,theta,t,params)
    deV = dT @ V
    d2eV = d2T @ V
    d2V = 30*(2*dq + d2q*p.T) + delta*d2eV - np.exp(-delta*(T @ V)/phi_bar)*delta*d2eV + np.exp(-delta*(T @ V)/phi_bar)*delta*delta*deV*deV/phi_bar
    return d2V.T

def T_s(p,P,s,q,theta,t,params):
    a,b,alpha,beta,gamma,sigma = theta
    delta,f,J,mu,nmax,S,upsilon_r = params
    uq =  (q*upsilon_r).flatten()
    Ns = S[:,1]
    row_terminal = np.arange(len(Ns))[Ns==nmax]
    row = np.arange(len(Ns))[Ns<nmax]
    column_terminal = np.arange(len(Ns))[Ns==nmax]
    column_no = np.arange(len(Ns))[Ns<nmax]
    column_bad = np.arange(len(Ns))[Ns<nmax]+Ns[Ns<nmax]+1
    column_good = np.arange(len(Ns))[Ns<nmax]+Ns[Ns<nmax]+2
    rows =  np.hstack((row_terminal,row,row,row))
    columns = np.hstack((column_terminal,column_no,column_bad,column_good))
    entry_terminal = np.ones(len(np.arange(len(Ns))[Ns==nmax]))
    entry_no = 1-uq[Ns<nmax]
    entry_bad = (uq*(1 - (a+Ns)/(a+b+Ns) ))[Ns<nmax]
    entry_good = (uq*((a+Ns)/(a+b+Ns)))[Ns<nmax]
    entries = np.hstack((entry_terminal,entry_no,entry_bad,entry_good))
    return csr_matrix((entries, (rows,columns)), shape = (len(Ns), len(Ns)))

def dT_s(p,P,s,dq,theta,t,params):
    a,b,alpha,beta,gamma,sigma = theta
    delta,f,J,mu,nmax,S,upsilon_r = params  
    #uq =  (1-(1-q*upsilon_r)**mu).flatten()
    duq = (dq*upsilon_r).flatten()
    Ns = S[:,1]
    row = np.arange(len(Ns))[Ns<nmax]
    column_no = np.arange(len(Ns))[Ns<nmax]
    column_bad = np.arange(len(Ns))[Ns<nmax]+Ns[Ns<nmax]+1
    column_good = np.arange(len(Ns))[Ns<nmax]+Ns[Ns<nmax]+2
    rows =  np.hstack((row,row,row))
    columns = np.hstack((column_no,column_bad,column_good))
    entry_no = -duq[Ns<nmax]
    entry_bad = (duq*(1 - (a+Ns)/(a+b+Ns) ))[Ns<nmax]
    entry_good = (duq*((a+Ns)/(a+b+Ns)))[Ns<nmax]
    entries = np.hstack((entry_no,entry_bad,entry_good))
    return csr_matrix((entries, (rows,columns)), shape = (len(Ns), len(Ns)))

def d2T_s(p,P,s,d2q,theta,t,params):
    a,b,alpha,beta,gamma,sigma = theta
    delta,f,J,mu,nmax,S,upsilon_r = params  
    d2uq = (d2q*upsilon_r).flatten()
    Ns = S[:,1]
    row = np.arange(len(Ns))[Ns<nmax]
    column_no = np.arange(len(Ns))[Ns<nmax]
    column_bad = np.arange(len(Ns))[Ns<nmax]+Ns[Ns<nmax]+1
    column_good = np.arange(len(Ns))[Ns<nmax]+Ns[Ns<nmax]+2
    rows =  np.hstack((row,row,row))
    columns = np.hstack((column_no,column_bad,column_good))
    entry_no = -d2uq[Ns<nmax]
    entry_bad = (d2uq*(1 - (a+Ns)/(a+b+Ns) ))[Ns<nmax]
    entry_good = (d2uq*((a+Ns)/(a+b+Ns)))[Ns<nmax]
    entries = np.hstack((entry_no,entry_bad,entry_good))
    return csr_matrix((entries, (rows,columns)), shape = (len(Ns), len(Ns)))

def F_s(p,P,s,q,chi,lamb,theta,t,params):
    a,b,alpha,beta,gamma,sigma = theta
    delta,f,J,mu,nmax,S,upsilon_r = params
    lamb1, lamb2, lamb3, lamb4 = lamb    
    uq = (q*upsilon_r).flatten()
    Ns = np.append(S[:,1],np.array([np.nan,np.nan,np.nan,np.nan]))
    row_terminal = np.arange(len(Ns))[Ns==nmax]
    row = np.arange(len(Ns))[Ns<nmax]
    row_entry1 = np.repeat(np.arange(len(Ns))[np.isnan(Ns)][0],8)
    row_entry2 = np.repeat(np.arange(len(Ns))[np.isnan(Ns)][1],8)
    row_entry3 = np.repeat(np.arange(len(Ns))[np.isnan(Ns)][2],8)
    row_entry4 = np.repeat(np.arange(len(Ns))[np.isnan(Ns)][3],8)
    row_exit = np.arange(len(Ns))[:-4]
    column_terminal = np.arange(len(Ns))[Ns==nmax]
    column_no = np.arange(len(Ns))[Ns<nmax]
    column_bad = np.arange(len(Ns))[Ns<nmax]+Ns[Ns<nmax]+1
    column_good = np.arange(len(Ns))[Ns<nmax]+Ns[Ns<nmax]+2
    column_entry = np.arange(len(Ns))[(Ns==0) | (np.isnan(Ns))]
    column_exit1 = np.repeat(np.arange(len(Ns))[np.isnan(Ns)][0],len(Ns)-4)  
    column_exit2 = np.repeat(np.arange(len(Ns))[np.isnan(Ns)][1],len(Ns)-4)
    column_exit3 = np.repeat(np.arange(len(Ns))[np.isnan(Ns)][2],len(Ns)-4)
    column_exit4 = np.repeat(np.arange(len(Ns))[np.isnan(Ns)][3],len(Ns)-4)
    rows = np.hstack((row_terminal,row,row,row,row_entry1,row_entry2,row_entry3,row_entry4,row_exit,row_exit,row_exit,row_exit))
    columns = np.hstack((column_terminal,column_no,column_bad,column_good,column_entry,column_entry,column_entry,column_entry,column_exit1,column_exit2,column_exit3,column_exit4))
    entry_terminal = ((1-chi)*np.ones(len(Ns)-4))[Ns[:-4]==nmax]
    entry_no = ((1-chi)*(1-uq))[Ns[:-4]<nmax]
    entry_bad = ((1-chi)*uq*(1 - (a+Ns[:-4])/(a+b+Ns[:-4]) ))[Ns[:-4]<nmax]
    entry_good = ((1-chi)*uq*((a+Ns[:-4])/(a+b+Ns[:-4])))[Ns[:-4]<nmax]   
    entry_entry1 = [lamb1,0,0,0,1-lamb1,0,0,0]
    entry_entry2 = [0,lamb2,0,0,0,1-lamb2,0,0]
    entry_entry3 = [0,0,lamb3,0,0,0,1-lamb3,0]
    entry_entry4 = [0,0,0,lamb4,0,0,0,1-lamb4]
    entry_exit = chi/4
    entries = np.hstack((entry_terminal,entry_no,entry_bad,entry_good,entry_entry1,entry_entry2,entry_entry3,entry_entry4,entry_exit,entry_exit,entry_exit,entry_exit))
    return csr_matrix((entries, (rows,columns)), shape = (len(Ns), len(Ns)))

def solver(theta,c,guess,t,Sub,tol,params):
    a,b,alpha,beta,gamma,sigma = theta
    kappa1, kappa2, kappa3, kappa4 = c[:4]
    phi1, phi2, phi3, phi4 = c[4:]
    phi_bar = np.array([np.repeat([phi1,phi2,phi3,phi4],231)]).T
    P_init,s_init,V_init = guess
    delta,f,J,mu,nmax,S,upsilon_r = params
    V_old = V_init
    P_old = P_init
    s_old = s_init
    dV = 1e10
    change = 1e10
    diff = 1e10
    dP = 1e10
    while diff>tol:
        if (change > 0.01):
            P0 = P_init
            P1 = P0 - dV_s(P0,P_old,s_old,V_old,theta,phi_bar,t,params)/d2V_s(P0,P_old,s_old,V_old,theta,phi_bar,t,params)
            P1 = np.where(np.isnan(P1) == True,P_old,np.where((P1<0),0,np.where((P1>1000),1000,P1)))
                
            dP = np.max(np.abs(P1 - P0))
            print('dP:',dP)
            #P0 = P1
            P_new = P1
            dVprim = dV
        else:
            P_new = P_old
        q_new = q_s(P_new,P_new,s_old,theta,t,params)
        T = T_s(P_new,P_new,s_old,q_new,theta,t,params)
        eV = T @ V_old
        V_new = 30*(q_new*P_new.T) + Sub + delta*eV - (phi_bar - np.exp(-delta*eV/phi_bar)*phi_bar)
        eV = T @ V_new
        chi = np.exp(-delta*eV/phi_bar).flatten()
        lamb = (1-np.exp(-delta*V_new.reshape((231,4),order='F')[0,:]/[kappa1,kappa2,kappa3,kappa4]))
        F = F_s(P_new,P_new,s_old,q_new,chi,lamb,theta,t,params)
        s_new = (np.array([np.append(s_old,np.array([J/4-s_old[0,:231].sum(),J/4-s_old[0,231:462].sum(),J/4-s_old[0,462:693].sum(),J/4-s_old[0,693:].sum()]))])@ F)[:1,:-4]
        while np.max(np.abs(s_new - s_old))>10e-3:
            s_old = s_new
            s_new = (np.array([np.append(s_old,np.array([J/4-s_old[0,:231].sum(),J/4-s_old[0,231:462].sum(),J/4-s_old[0,462:693].sum(),J/4-s_old[0,693:].sum()]))])@ F)[:1,:-4]   
        dV = np.max(np.abs(V_new - V_old))
        dP = np.max(np.abs(P_new - P_old))
        ds = np.max(np.abs(s_new - s_old))
        print('dV:',dV)
        print('dP:',dP)
        print('ds:',ds)
        diff = max(dV,dP,ds)
        change = (dVprim - dV)/dVprim
        V_old = V_new
        P_old = P_new
        s_old = s_new
    return [V_new,s_new,P_new,chi,lamb]
