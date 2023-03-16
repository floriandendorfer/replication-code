# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 09:20:40 2023

@author: Florian Dendorfer
"""

import numpy as np

################################  Parameters  #################################

alpha = -.0105
beta = np.array([-13.9974,-13.9974+0.1884,-13.9974+0.3084,-13.9974+0.4544])
gamma = 3.4554
a = 7.8516
b = 4.3545
kappa_bar = 25106
phi_bar = 18535
delta = .99
f = .142
mu = 20000
upsilon_a = 0.019
nmax = 20
J = 2000

theta = [a,b,alpha,beta,gamma]
c = [kappa_bar,phi_bar]

################################  Type space  #################################

from functions import cartesian_product

S = cartesian_product(np.arange(nmax+1),np.arange(nmax+1))
S = S[S[:,0]<=S[:,1]]
S=S[np.lexsort((S[:,0],S[:,1]))]
S=np.repeat(S[:,:,np.newaxis],4,axis=2)

upsilon_r = 0.2938*np.log(S[:,:1,0]+1)

###############################  Set of prices  ###############################

prices = np.array([np.linspace(0,300,301)])

##############################  Load functions  ###############################

    # Load utility function
from functions import U

    # Load demand function
from functions import q

    # Load profit function
from functions import pi
pi(P_init,s_init,theta,0,S,f,mu,prices)

np.diagonal(U(1,theta,0,S,f),axis1=0,axis2=1).reshape((len(S)*4,1))

U(P_init,theta,0,S,f)