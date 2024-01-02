# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 09:20:42 2023

@author: Florian Dendorfer
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import expit
import shelve 
import time

#################################  Functions  #################################

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def U(p,theta,t,params):
    a,b,alpha,beta,gamma=theta
    delta,f,J,mu,nmax,S,upsilon_r = params
    return gamma*( (a+S[:,0:1])/(a+b+S[:,1:2]) ) + S[:,2:] @ beta + alpha*(p*(1+f)-t)
    
def ccp_s(p,P,s,theta,t,params):
    a,b,alpha,beta,gamma = theta
    delta,f,J,mu,nmax,S,upsilon_r = params
    U_j = np.array([np.diagonal(U(p,theta,t,params),axis1=0,axis2=1)]).T
    U_all = np.array([np.diagonal(U(P,theta,t,params),axis1=0,axis2=1)]).T 
    ccp = np.exp(U_j)/(1 + np.exp(U_j) - np.tile(np.exp(U_all),(1,U_j.shape[1])) + float(s @ np.exp(U_all)) )
    return ccp
 
def q_s(p,P,s,theta,t,params):
    a,b,alpha,beta,gamma = theta
    delta,f,J,mu,nmax,S,upsilon_r = params
    ccp = ccp_s(p,P,s,theta,t,params)
    q = 1 - np.exp(-mu*ccp)
    return q

def dq_s(p,P,s,theta,t,params):
    a,b,alpha,beta,gamma = theta
    delta,f,J,mu,nmax,S,upsilon_r = params
    ccp = ccp_s(p,P,s,theta,t,params)
    return mu*np.exp(-mu*ccp)*ccp*(1-ccp)*alpha*(1+f)
    
def d2q_s(p,P,s,theta,t,params):
    a,b,alpha,beta,gamma = theta
    delta,f,J,mu,nmax,S,upsilon_r = params
    ccp = ccp_s(p,P,s,theta,t,params)
    return mu*( -mu*np.exp(-mu*ccp)*ccp*ccp*(1-ccp)*(1-ccp)*alpha*(1+f)
               + np.exp(-mu*ccp)*ccp*(1-ccp)*(1-ccp)*alpha*(1+f)
               - np.exp(-mu*ccp)*ccp*(1-ccp)*(1-ccp)*alpha*(1+f)
               )*alpha*(1+f)
    
def dV_s(p,P,s,V,theta,phi_bar,t,params):
    a,b,alpha,beta,gamma=theta
    delta,f,J,mu,nmax,S,upsilon_r = params
    q = q_s(p,P,s,theta,t,params)
    dq = dq_s(p,P,s,theta,t,params)
    dT = dT_s(dq,theta,params)
    T = T_s(q,theta,params)
    deV = dT @ V
    dV = 30*(q + dq*p.T) + delta*deV - np.exp(-delta*(T @ V)/phi_bar)*delta*deV
    return dV.T

def d2V_s(p,P,s,V,theta,phi_bar,t,params):
    a,b,alpha,beta,gamma=theta
    delta,f,J,mu,nmax,S,upsilon_r = params
    q = q_s(p,P,s,theta,t,params)
    dq = dq_s(p,P,s,theta,t,params)
    d2q = d2q_s(p,P,s,theta,t,params)
    T = T_s(q,theta,params)
    dT = dT_s(dq,theta,params)
    d2T = d2T_s(d2q,theta,params)
    deV = dT @ V
    d2eV = d2T @ V
    d2V = 30*(2*dq + d2q*p.T) + delta*d2eV - np.exp(-delta*(T @ V)/phi_bar)*delta*d2eV + np.exp(-delta*(T @ V)/phi_bar)*delta*delta*deV*deV/phi_bar
    return d2V.T

def T_s(q,theta,params):
    a,b,alpha,beta,gamma=theta
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

def dT_s(dq,theta,params):
    a,b,alpha,beta,gamma=theta
    delta,f,J,mu,nmax,S,upsilon_r = params  
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
    dT = csr_matrix((entries, (rows,columns)), shape = (len(Ns), len(Ns)))
    return dT

def d2T_s(d2q,theta,params):
    a,b,alpha,beta,gamma=theta
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
    d2T = csr_matrix((entries, (rows,columns)), shape = (len(Ns), len(Ns)))
    return d2T

def F_s(q,chi,lamb,theta,params):
    a,b,alpha,beta,gamma=theta
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

def solver(theta,c,guess,tol,params):
    a,b,alpha,beta,gamma = theta
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
    while diff>tol:
        if (change > 0.1):
            dP = 1e10
            P0 = P_old
            while dP>.1:
                P1 = P0 - dV_s(P0,P_old,s_old,V_old,theta,phi_bar,0,params)/d2V_s(P0,P_old,s_old,V_old,theta,phi_bar,0,params)
                P1 = np.where(np.isnan(P1) == True,P_old,np.where((P1<0),0,np.where((P1>1000),1000,P1)))
                dP = np.max(np.abs(P1 - P0))
                P0 = P1
            P_new = P1
            dVprim = dV
        else:
            P_new = P_old
        q_new = q_s(P_new,P_new,s_old,theta,0,params)
        T = T_s(q_new,theta,params)
        eV = T @ V_old
        V_new = 30*(q_new*P_new.T) + delta*eV - (1 - np.exp(-delta*eV/phi_bar))*phi_bar
        eV = T @ V_new
        chi = np.exp(-delta*eV/phi_bar).flatten()
        lamb = (1-np.exp(-delta*V_new.reshape((231,4),order='F')[0,:]/[kappa1,kappa2,kappa3,kappa4]))
        F = F_s(q_new,chi,lamb,theta,params)
        s_new = (np.array([np.append(s_old,np.array([J/4-s_old[0,:231].sum(),J/4-s_old[0,231:462].sum(),J/4-s_old[0,462:693].sum(),J/4-s_old[0,693:].sum()]))])@ F)[:1,:-4]
        while np.max(np.abs(s_new - s_old))>1e-2:
            s_old = s_new
            s_new = (np.array([np.append(s_old,np.array([J/4-s_old[0,:231].sum(),J/4-s_old[0,231:462].sum(),J/4-s_old[0,462:693].sum(),J/4-s_old[0,693:].sum()]))])@ F)[:1,:-4]
        dV = np.max(np.abs(V_new - V_old))
        dP = np.max(np.abs(P_new - P_old))
        ds = np.max(np.abs(s_new - s_old))
        diff = max(dV,dP,ds)
        change = (dVprim - dV)/dVprim
        V_old = V_new
        P_old = P_new
        s_old = s_new
        
    return [V_new,s_new,P_new,chi,lamb]

def xi(omicron,adata,params):
    delta,f,J,mu,nmax,S,upsilon_r = params
    psi,iota,alpha = omicron[:3]
    beta = np.array([omicron[3:7]]).T
    gamma = omicron[7]
    U = gamma*( (expit(psi)*np.exp(iota)+adata['K'])/(np.exp(iota)+adata['N']) )  + (adata[['type 1','type 2','type 3','type 4']] @ beta).iloc[:,0] + adata['p']*(1+f)*alpha  
    return np.array([(np.log(adata['share']) - np.log(adata['share_0']) - U).to_numpy()]).T
    
def dU(omicron,adata,params):   
    delta,f,J,mu,nmax,S,upsilon_r = params
    psi,iota,alpha = omicron[:3]
    gamma = omicron[7]
    dVdpsi = np.array([gamma*(np.exp(iota)/(np.exp(iota)+adata['N']))*expit(psi)*(1-expit(psi))]).T
    dVdiota = np.array([gamma*(np.exp(iota)/(np.exp(iota)+adata['N'])**2)*( expit(psi)*adata['N']-adata['K'])]).T
    dVdalpha = np.array([(1+f)*adata['p']]).T
    dVdbeta = adata[['type 1','type 2','type 3','type 4']]
    dVdgamma = np.array([(expit(psi)*np.exp(iota)+adata['K'])/(np.exp(iota)+adata['N'])]).T
    return np.hstack((dVdpsi,dVdiota,dVdalpha,dVdbeta,dVdgamma))

def Z(omicron,adata,params):   
    delta,f,J,mu,nmax,S,upsilon_r = params
    psi,iota,alpha = omicron[:3]
    ZN = adata[['N']]
    ZK = adata[['K']]
    ZNK = adata[['r']]
    ZP = np.array([(1+f)*adata['p']]).T
    Ztype = adata[['type 1','type 2','type 3','type 4']]
    return np.hstack((ZN,ZK,ZP,Ztype,ZNK))

def O(omicron,adata,W,params):
    xi_hat = xi(omicron,adata,params)
    Z_hat = Z(omicron,adata,params)
    value = (( ((Z_hat.T @ xi_hat).T ) @ W) @ ((Z_hat.T @ xi_hat) ) )[0,0]/(len(adata)*len(adata))
    print(value)
    return value

def dO(omicron,adata,W,params):
    xi_hat = xi(omicron,adata,params)
    Z_hat = Z(omicron,adata,params)
    dU_hat = dU(omicron,adata,params)
    return +2*(( (( (Z_hat.T @ (-dU_hat)) ).T ) @ W) @ ((Z_hat.T @ xi_hat) ) )[:,0]/(len(adata)*len(adata))

def start_values(*values):
    start_values.values = values or start_values.values
    return start_values.values

def likelihood(k,theta,tol,s_d,params):
    delta,f,J,mu,nmax,S,upsilon_r = params
    a,b,alpha = theta[:3]
    beta = theta[3]
    gamma = theta[4]
    shelve_file = shelve.open("guess") 
    guess = shelve_file['guess']
    shelve_file.close()    
    V_star,s_star,P_star,chi_star,lamb_star = solver([a,b,alpha,beta,gamma],np.exp(k),guess,tol,params)
    ll = np.hstack((np.array([s_d[(s_d>0) & (s_star>0)]]),np.array([[J/4-s_d[0,:231].sum()],[J/4-s_d[0:,231:462].sum()],[J/4-s_d[0,462:693].sum()],[J/4-s_d[0,693:].sum()]]).T)) @ np.log(np.hstack((np.array([s_star[(s_d>0) & (s_star>0)]]),np.array([[J/4-s_star[0,:231].sum()],[J/4-s_star[0,231:462].sum()],[J/4-s_star[0,462:693].sum()],[J/4-s_star[0,693:].sum()]]).T))/J).T
    shelve_file = shelve.open("gfg") 
    shelve_file['guess'] = [P_star,s_star,V_star] 
    shelve_file.close() 
    return -ll.sum()

def simulation1(theta,c,sol,t,Sub,It,params):
    a,b,alpha,beta,gamma = theta
    kappa1, kappa2, kappa3, kappa4, phi1, phi2, phi3, phi4 = c
    phi_bar = np.array([np.repeat([phi1,phi2,phi3,phi4],231)]).T
    delta,f,J,mu,nmax,S,upsilon_r = params
    P_old,s_old,V_old = sol
    W = np.array([])
    CS = np.array([])
    PS = np.array([])
    GS = np.array([])
    for it in range(It):
        dP = 1e10
        P0 = P_old
        while dP>1e-1:
            P1 = P0 - dV_s(P0,P_old,s_old,V_old,theta,phi_bar,t,params)/d2V_s(P0,P_old,s_old,V_old,theta,phi_bar,t,params)
            P1 = np.where(np.isnan(P1) == True,P_old,np.where((P1<0),0,np.where((P1>1000),1000,P1)))
            dP = np.max(np.abs(P1 - P0))
            P0 = P1
        P_new = P1
        q_new = q_s(P_new,P_new,s_old,theta,t,params)
        T = T_s(q_new,theta,params)
        eV = T @ V_old
        V_new = 30*(q_new*P_new.T) + Sub + delta*eV - (1 - np.exp(-delta*eV/phi_bar))*phi_bar
        eV = T @ V_new
        chi = np.exp(-delta*eV/phi_bar).flatten()
        lamb = (1-np.exp(-delta*V_new.reshape((231,4),order='F')[0,:]/[kappa1,kappa2,kappa3,kappa4]))
        F = F_s(q_new,chi,lamb,theta,params)
        s_new = (np.array([np.append(s_old,np.array([J/4-s_old[0,:231].sum(),J/4-s_old[0,231:462].sum(),J/4-s_old[0,462:693].sum(),J/4-s_old[0,693:].sum()]))])@ F)[:1,:-4]        
        lamb = np.array([np.repeat(lamb,231)]).T
        chi = np.array([chi]).T
        T = T_s(q_new,theta,params)
        eV_in = T @ V_new
        eV_out = V_new.reshape((231,4),order='F')[0,:]
        cs = -(s_new @ q_new)*30*np.log(1 + (s_new @ np.array([np.diagonal(np.exp(U(P_new,theta,t,params)))]).T) )/alpha
        ps_in = (s_new.T * (  (1+f)* (np.array([np.diagonal(30*q_s(P_new,P_new,s_new,theta,t,params)*P_new)]).T) + Sub - (np.array([np.repeat(c[4:],231)]).T - chi * (delta*eV_in + np.array([np.repeat(c[4:],231)]).T)) ) ).sum()
        ps1_out = -(J/4-s_old[0,:231].sum())*(c[0] - (1-lamb[0])*(delta*eV_out[0] + c[0]))
        ps2_out = -(J/4-s_old[0,231:462].sum())*(c[1] - (1-lamb[231])*(delta*eV_out[1] + c[1]))
        ps3_out = -(J/4-s_old[0,462:693].sum())*(c[2] - (1-lamb[462])*(delta*eV_out[2] + c[2]))
        ps4_out = -(J/4-s_old[0,693:].sum())*(c[3] - (1-lamb[693])*(delta*eV_out[3] + c[3]))
        gs = -((30*s_new*q_s(P_new,P_new,s_new,theta,t,params).T) @ t + s_new @ Sub)
        CS = np.append(CS,cs)
        PS = np.append(PS,(ps_in + ps1_out + ps2_out + ps3_out + ps4_out))
        GS = np.append(GS,gs)
        W = np.append(W,cs + ps_in + ps1_out + ps2_out + ps3_out + ps4_out + gs)
        V_old = V_new
        P_old = P_new
        s_old = s_new
    return [W,CS,PS,GS,P_new,s_new,V_new]

def simulation2(theta,c,sol,t,Sub,It,params):
    a,b,alpha,beta,gamma = theta
    kappa1, kappa2, kappa3, kappa4, phi1, phi2, phi3, phi4 = c
    phi_bar = np.array([np.repeat([phi1,phi2,phi3,phi4],231)]).T
    delta,f,J,mu,nmax,S,upsilon_r = params
    P_init,s_init,V_init = sol
    P_old,s_old,V_old = sol
    W = np.array([])
    CS = np.array([])
    PS = np.array([])
    GS = np.array([])
    for it in range(It):
        dP = 1e10
        P0 = P_old
        while dP>1e-1:
            P1 = P0 - dV_s(P0,P_old,s_old,V_old,theta,phi_bar,t,params)/d2V_s(P0,P_old,s_old,V_old,theta,phi_bar,t,params)
            P1 = np.where(np.isnan(P1) == True,P_old,np.where((P1<0),0,np.where((P1>1000),1000,P1)))
            dP = np.max(np.abs(P1 - P0))
            P0 = P1
        P_new = P1
        q_new = q_s(P_new,P_new,s_old,theta,t,params)
        T = T_s(q_new,theta,params)
        eV = T @ V_old
        V_new = 30*(q_s(P_init,P_init,s_init,theta,np.zeros((S.shape[0],1)),params)*P_init.T) + Sub + delta*eV - (1 - np.exp(-delta*eV/phi_bar))*phi_bar
        eV = T @ V_new
        chi = np.exp(-delta*eV/phi_bar).flatten()
        lamb = (1-np.exp(-delta*V_new.reshape((231,4),order='F')[0,:]/[kappa1,kappa2,kappa3,kappa4]))
        F = F_s(q_new,chi,lamb,theta,params)
        s_new = (np.array([np.append(s_old,np.array([J/4-s_old[0,:231].sum(),J/4-s_old[0,231:462].sum(),J/4-s_old[0,462:693].sum(),J/4-s_old[0,693:].sum()]))])@ F)[:1,:-4]        
        lamb = np.array([np.repeat(lamb,231)]).T
        chi = np.array([chi]).T
        T = T_s(q_new,theta,params)
        eV_in = T @ V_new
        eV_out = V_new.reshape((231,4),order='F')[0,:]
        cs = -(s_new @ q_new)*30*np.log(1 + (s_new @ np.array([np.diagonal(np.exp(U(P_new,theta,t,params)))]).T) )/alpha
        ps_in = (s_new.T * ( f*(np.array([np.diagonal(30*q_s(P_new,P_new,s_new,theta,t,params)*P_new)]).T) + (np.array([np.diagonal(30*q_s(P_init,P_init,s_init,theta,np.zeros((S.shape[0],1)),params)*P_init)]).T) + Sub - (np.array([np.repeat(c[4:],231)]).T - chi * (delta*eV_in + np.array([np.repeat(c[4:],231)]).T)) ) ).sum()
        ps1_out = -(J/4-s_old[0,:231].sum())*(c[0] - (1-lamb[0])*(delta*eV_out[0] + c[0]))
        ps2_out = -(J/4-s_old[0,231:462].sum())*(c[1] - (1-lamb[231])*(delta*eV_out[1] + c[1]))
        ps3_out = -(J/4-s_old[0,462:693].sum())*(c[2] - (1-lamb[462])*(delta*eV_out[2] + c[2]))
        ps4_out = -(J/4-s_old[0,693:].sum())*(c[3] - (1-lamb[693])*(delta*eV_out[3] + c[3]))
        gs = -((30*s_new*q_s(P_new,P_new,s_new,theta,t,params).T) @ t + s_new @ (Sub - 30*(q_s(P_new,P_new,s_new,theta,t,params).T*P_new - q_s(P_init,P_init,s_init,theta,np.zeros((S.shape[0],1)),params).T*P_init).T))
        CS = np.append(CS,cs)
        PS = np.append(PS,(ps_in + ps1_out + ps2_out + ps3_out + ps4_out))
        GS = np.append(GS,gs)
        W = np.append(W,cs + ps_in + ps1_out + ps2_out + ps3_out + ps4_out + gs)
        V_old = V_new
        P_old = P_new
        s_old = s_new
    return [W,CS,PS,GS,P_new,s_new,V_new]


def Sub_prim(Sub,theta,c,sol,It,params):
    delta,f,J,mu,nmax,S,upsilon_r = params
    x = np.zeros((S.shape[0],1))
    x[[0,231,462,693],:] = np.array([Sub]).T
    Sub = x
    W,CS,PS,GS,P,s,V = simulation1(theta,c,sol,np.zeros((S.shape[0],1)),Sub,It,params)
    W = (np.array([W]) @ np.array([delta**np.arange(0,It)]).T)[0][0]
    print(W)
    if np.isnan(W):
        return -np.infty
    else:
        return -W

def t_prim(t,theta,c,sol,Sub,It,params):
    delta,f,J,mu,nmax,S,upsilon_r = params
    y = np.zeros((S.shape[0],1))
    y[[0,231,462,693],:] = np.array([Sub]).T
    Sub = y
    x = np.zeros((S.shape[0],1))
    x[[0,231,462,693],:] = np.array([t]).T
    t = x
    W,CS,PS,GS,P,s,V = simulation2(theta,c,sol,t,Sub,It,params)
    W = (np.array([W]) @ np.array([delta**np.arange(0,It)]).T)[0][0]
    print(W)
    if np.isnan(W):
        return -np.infty
    else:
        return -W