# Replication code

## Parameter

|  | name |            |  value |
  | ---: | ---: | :---------: | :------: |
  | demand (<code>theta</code>) | price | $\alpha$ | -0.0105 |
  || type 1 | $\beta_1$ | -11.1007 |
  || type 2 | $\beta_2$ | -10.7004 |
  || type 3 | $\beta_3$ | -10.5486 |
  || type 4 | $\beta_4$ | -10.3437 |
  || quality | $\gamma$ | 3.79019 |
  || prior | $a$ | 7.9686 |
  ||  | $b$ | 1.4726 |
  | supply (<code>c</code>) | mean entry cost (type 1) | $\bar \kappa_1$ | 37,130 |
  |  | mean entry cost (type 2) | $\bar \kappa_2$ | 237,883 |
  |  | mean entry cost (type 3) | $\bar \kappa_3$ | 182,152 |
  |  | mean entry cost (type 4) | $\bar \kappa_4$ | 356,233 |
  || mean fixed cost (type 1) | $\bar \phi_1$ | 2,468 |
  | | mean fixed cost (type 2) | $\bar \phi_2$ | 3,037 |
  || mean fixed cost (type 3) | $\bar \phi_3$ | 3,367 |
  || mean fixed cost (type 4) | $\bar \phi_4$ | 3,839 |
  | other (<code>params</code>) | discount factor | $\delta$ | 0.995 |
  |  | revenue fee | $f$ | 0.142 |
  |  | review prob. | $\upsilon_r$ | 0.7041 |
  |  | max. no. of reviews | $\bar N$ | 20 |
  |  | arrival rate | $\mu$ | 10,000 |
  |  | max. no. of listings | $J$ | 10,000 |

## Demand

Function <code>U_s(p,theta,t,params)</code> characterizes a guests's indirect **utility** of renting a property in state $x=(N,K,j)$, where $j = 1,2,3,4$ is the property's (observed) **type**.

  $$U_x = \gamma\frac{a + K(x)}{a + b + N(x)} + \sum_{j'}\beta_j\mathbb{1}(j' = j) + \alpha ((1+f)p- t) + \epsilon = u(p,x) + \epsilon$$
  
$p$ is the daily rental rate of the listing; $t$ is the counterfactual per-unit subsidy. For the moment, we set $t$ equal to zero.

**Unobserved quality**: The unobserved quality $\omega$ is unknown to guests and hosts. However, $\omega$ is known to be iid $Beta(a,b)$ distributed. After observing the number of good reviews $K$ and bad reviews $N-K$ agents form an expectation about the unobserved quality, $E[\omega|N,K]$.</li>

$\epsilon$ is iid T1EV extreme value distributed.

Function <code>ccp_s(p,P,s,theta,t,params)</code> characterizes the probability that a guest ***intends*** to book the property at rate $p$ provided that all remaining hosts set their prices according to $P(x)$.

$$ccp(p,x) = \frac{\exp(u(p,x))}{1+\sum_xs(x)\exp(u(P(x),x))}$$

**State distribution**: $s(x)$ pins down the number of properties in each state. For later use, we also work out the first-order (<code>dccp_s(p,P,s,theta,t,params)</code>) and second-order (<code>d2ccp_s(p,P,s,theta,t,params)</code>) derivatives of $ccp(p,x)$ with respect to $p$.

$$ccp'(p,x) = ccp(p,x)(1 - ccp(p,x))\alpha(1+f) $$

$$ccp''(p,x) = ccp(p,x)(ccp(p,x)^2 - ccp(p,x))\alpha^2(1+f)^2 $$

**Demand**: The number of arriving guests is $Poisson(\mu)$ distributed. Function <code>q_s(p,P,s,theta,t,params)</code> characterizes the probability that at least one of these consumers books the property, again assuming its rental rate is $p$ while everyone else follows the pricing rule $P(x)$.

$$q(p,x) = 1 - \exp(-\mu \cdot ccp(p,x))$$

Function <code>d2q_s(p,P,s,theta,t,params)</code> and function <code>d2q_s(p,P,s,theta,t,params)</code> describe the first- and second-order derivatives of $q(p,x)$ with respect to $p$.

  $$q'(p,x) = \mu\exp(-\mu \cdot ccp(p,x))ccp'(p,x)$$
  $$q''(p,x) = \mu\exp(-\mu \cdot ccp(p,x))(ccp''(p,x)-\mu\cdot ccp'(p,x))$$

Strictly speaking, $q_s$ is the ***daily*** booking probability. As a **time period** in the model is a 4-week interval ("month"), we interpret $q_s$ as the monthly **occupancy rate**. 

## State Transitions

If a property is booked ($q(p,x) = 1$), $x$ changes with probability $\upsilon_r = 70.41$% between periods. Conditional on being booked, it receives a good review ($\Delta N = 1, \Delta K = 1$) with probability $\frac{a+K(x)}{a+b+N(x)}$. Conditional on being booked, it receives a bad review ($\Delta N = 1, \Delta K = 0$) with probability $\left(1-\frac{a + K(x)}{a+b+N(x)}\right)$. The **probability of getting a good review**  and the **probability of getting a bad review** are $\rho^g(p,x)$ and $\rho^b(p,x)$ respectively. States where $N=20$ are **terminal** and the probability of getting a review is zero.

$$\rho^g(p,x) = \upsilon_rq(p,x)\frac{a+K(x)}{a+b+N(x)}$$
$$\rho^b(p,x) = \upsilon_rq(p,x)\left(1-\frac{a + K(x)}{a+b+N(x)}\right)$$

Accordingly, the probability $\rho^0(p,x)$ of getting no review is $1-\rho^g(p,x)-\rho^b(p,x)$. States are arranged in increasing order of type $j$ and, for a given type, in increasing order of $N$ and, for a given $N$, in increasing order of $K$. $S$ is the **state space**. Note: $S$ is in <code>params</code>.

$$ S = \begin{bmatrix} 
1 & 0 & 0 & 0 & 0 & 0 \\ 
1 & 0 & 1 & 0 & 0 & 0 \\
2 & 1 & 1 & 0 & 0 & 0 \\
2 & 0 & 2 & 0 & 0 & 0 \\
2 & 1 & 2 & 0 & 0 & 0 \\
3 & 2 & 2 & 0 & 0 & 0 \\
... & ... & ... & ... & ... & ... \\ 
11 & 17 & 0 & 0 & 1 & 0 \\
12 & 17 & 0 & 0 & 1 & 0 \\
13 & 17 & 0 & 0 & 1 & 0 \\
... & ... & ... & ... & ... & ... \\ 
19 & 20 & 0 & 0 & 0 & 1 \\
20 & 20 & 0 & 0 & 0 & 1
\end{bmatrix} $$

Function  <code>dT_s(dq,theta,params)</code> stores the **transition matrix** $T(p,x)$. It turns out that the way states are ordered the number of zeros between $\rho^0(p,x)$ and $\rho^g(p,x)$ is $N$.  

|  | $(0,0,1)$ | $(0,1,1)$ | $(1,1,1)$ | $(0,2,1)$ | $(1,2,1)$ | $(2,2,1)$ | ... | $(20,20,4)$ | 
| :---: | :---: | :---------: | :------: | :------: | :------: | :------: | :------: | :------: |
| $(0,0,1)$ | $\rho^0_{(0,0,1)}$ | $\rho^g_{(0,0,1)}$ | $\rho^b_{(0,0,1)}$ | 0 | 0 | 0 | ... | 0 |
| $(0,1,1)$ | 0 | $\rho^0_{(0,1,1)}$ | 0 | $\rho^g_{(0,1,1)}$ | $\rho^b_{(0,1,1)}$ | 0 | ... | 0 |
| $(1,1,1)$ | 0 | 0 | $\rho^0_{(1,1,1)}$ | 0 | $\rho^g_{(1,1,1)}$ | $\rho^b_{(1,1,1)}$ | ... | 0 |
| $(0,2,1)$ | 0 | 0 | 0 | $\rho^0_{(0,2,1)}$ | 0 | 0 | ... | 0 |
| $(1,2,1)$ | 0 | 0 | 0 | 0 | $\rho^0_{(1,2,1)}$ | 0 | ... | 0 |
| $(2,2,1)$ | 0 | 0 | 0 | 0 | 0 | $\rho^0_{(2,2,1)}$ | ... | 0 |
| ... | ... | ... | ... | ... | ... | ... | ... | 0 |
| $(20,20,4)$ | 0 | 0 | 0 | 0 | 0 | 0 | ... | 1 |

Function <code>dT_s(q,theta,params)</code> and <code>d2T_s(q,theta,params)</code> store the first-order and second-order derivatives of $T(q)$ respectively. Notice:

$$\rho^{0\prime}(p,x) = -\upsilon_rq'(p,x)$$

$$\rho^{0\prime\prime}(p,x) = -\upsilon_rq''(p,x)$$

$$\rho^{g\prime}(p,x) = \upsilon_rq'(p,x)\left(\frac{a+K(x)}{a+b+N(x)}\right)$$

$$\rho^{g\prime\prime}(p,x) = \upsilon_rq''(p,x)\left(\frac{a+K(x)}{a+b+N(x)}\right)$$

$$\rho^{b\prime}(p,x) = \upsilon_rq'(p,x)\left(1-\frac{a+K(x)}{a+b+N(x)}\right)$$

$$\rho^{b\prime\prime}(p,x) = \upsilon_rq''(p,x)\left(1-\frac{a+K(x)}{a+b+N(x)}\right)$$

## Market Entry & Exit

Types are equally distributed in the host population, meaning 2,500 properties have a certain type. If a host is **inactive** and has not yet entered the market, they can do so at the start of the following month at **entry cost** $\kappa$ which is iid drawn from $Exponential(\bar \kappa_j)$, $j=1,2,3,4$. Let $\lambda_j$ denote the **entry rate**. 

$$ \lambda_j = 1-\exp(-\delta V((0,0,j))]\bar\kappa_j^{-1} ) $$

The expected, total entry costs of type-$\tilde j$ hosts in a given month is the number of inactive hosts $(J/4 - \sum_{x}\mathbb{1}(\tilde j = j)s(x))$ times $\mathbb{E}[\kappa_j|\phi_j\geq \delta V(0,0,j)]$.

$$ \text{Total entry costs} = \sum_{\tilde j}(J/4 - \sum_{x}\mathbb{1}(\tilde j = j)s(x))\left(\lambda_j\bar \kappa_j - (1-\lambda_j)\delta V((0,0,j))\right) $$

If a host is **active** they have entered the market. At the end of each month they have to pay the **operating cost** $\phi_j$ for the following month, regardless of whether the property is booked or not. $\phi_j$ is iid $Exponential(\bar \phi_j)$ distributed. Let $\chi(p,x)$ denote the **exit rate**.

$$ \chi(p,x) = \exp(-\delta \mathbb{E}_{x'}[V(x')|p,x]\bar\phi_j^{-1} ). $$

$x'$ denotes the state in the next month. Note that the host's expectation depends on $p$ because the property is likely to transition to a new state if it is booked.  

The expected, total operating costs of type-$\tilde j$ hosts in a given month are the number of active hosts $\sum_{x}\mathbb{1}(\tilde j = j)s(x)$ times $\mathbb{E}[\phi_j|\phi_j\leq \delta \mathbb{E}_{x'}[V(x')|p,x]]$ 

$$ \text{Total operating costs} = \sum_{x}s(x)\left((1-chi(p,x))\bar \kappa_j - chi(p,x)\delta \mathbb{E}_{x'}[V(x')|p,x]\right) $$ 

<code>F_s(p,P,s,q,chi,lamb,theta,t,params)</code> contains the **expanded transition matrix** F(p,x). It accommodate transitions from and to inactivity by expanding $T(p,x)$ by an additional state.

|  | $(0,0,1)$ | $(0,1,1)$ | $(1,1,1)$ | $(0,2,1)$ | $(1,2,1)$ | $(2,2,1)$ | ... | $(0,0,2)$ | ... | $(20,20,4)$ | $(20,20,4)$ |  
| :---: | :---: | :---------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| $(0,0,1)$ | $(1-\chi_{(0,0,1)})\rho^0_{(0,0,1)}$ | $(1-\chi_{(0,0,1)})\rho^g_{(0,0,1)}$ | $(1-\chi_{(0,0,1)})\rho^b_{(0,0,1)}$ | 0 | 0 | 0 | ... | ... | ... | 0 | $\chi_{(0,0,1)}$ |
| $(0,1,1)$ | 0 | $(1-\chi_{(0,1,1)})\rho^0_{(0,1,1)}$ | 0 | $(1-\chi_{(0,1,1)})\rho^g_{(0,1,1)}$ | $(1-\chi_{(0,1,1)})\rho^b_{(0,1,1)}$ | 0 | ... | ... | ... | 0 | $\chi_{(0,1,1)}$ |
| $(1,1,1)$ | 0 | 0 | $(1-\chi_{(1,1,1)})\rho^0_{(1,1,1)}$ | 0 | $(1-\chi_{(1,1,1)})\rho^g_{(1,1,1)}$ | $(1-\chi_{(1,1,1)})\rho^b_{(1,1,1)}$ | ... | ... | ... | 0 | $\chi_{(1,1,1)}$ |
| $(0,2,1)$ | 0 | 0 | 0 | $(1-\chi_{(1,2,1)})\rho^0_{(1,2,1)}$ | 0 | 0 | ... | ... | ... | 0 | $\chi_{(1,2,1)}$ |
| $(1,2,1)$ | 0 | 0 | 0 | 0 | $(1-\chi_{(1,2,1)})\rho^0_{(1,2,1)}$ | 0 | ... | ... | ... | 0 | $\chi_{(1,2,1)}$ |
| $(2,2,1)$ | 0 | 0 | 0 | 0 | 0 | $(1-\chi_{(2,2,1)})\rho^0_{(2,2,1)}$ | ... | ... | ... | 0 | $\chi_{(2,2,1)}$ |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| $(0,0,2)$ | 0 | 0 | 0 | 0 | 0 | 0 | ... | $(1-\chi_{(0,0,2)})\rho^0_{(0,0,2)}$ | ... | 0 | $\chi_{(0,0,2)}$ |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| $(20,20,4)$ | 0 | 0 | 0 | 0 | 0 | 0 | ... | ... | ... | $1 - \chi_{(20,20,4)}$ | $\chi_{(20,20,4)}$ |
| $\varnothing_1$ | $\lambda_1$ | 0 | 0 | 0 | 0 | 0 | ... | ... | ... | 0 | $1-\lambda_1$ |
| $\varnothing_2$ | 0 | 0 | 0 | 0 | 0 | 0 | ... | $\lambda_2$ | ... | 0 | $1-\lambda_2$ |
| $\varnothing_3$ | 0 | 0 | 0 | 0 | 0 | 0 | ... | ... | ... | 0 | $1-\lambda_3$ |
| $\varnothing_4$ | 0 | 0 | 0 | 0 | 0 | 0 | ... | ... | ... | 0 | $1-\lambda_4$ |

## Model Solution

<code>solver(theta,c,guess,t,tol,params)</code> finds a oblivious equilibrium of the model. <code>guess</code> contains starting values for the pricing function $P(x')$, the state distribution $s(x')$ and the value function $V(x').

  ### Pricing

Conditional on guess $V(x')$ and assuming s(x') competitors set their prices according to $P(x')$, a host operating a property in state $x$ maximizes $V(x)$ over $p$.

$$ V(p,x) = 30q(p,x)p - (1-\chi(p,x))\phi(x) + \delta T(p,x)V(x') $$

The FOC requires that $V'(p,x) = 0$. The first-order Taylor series approximation around $p_0$ is $V'(p,x) = V'(p_0,x) + V''(p_0,x)(p-p_0)$. We find $p$ by iterating $ p = p_0 + \frac{V'(p_0,x)}{V'(p_0,x)}$ until $|p-p_0| \leq 0.1$. 

<code>dV_s(p,P,s,V,theta,\phi_bar,t,params)</code> and <code>d2V_s(p,P,s,V,theta,\phi_bar,t,params)</code> store the first- and second-order derivative of $V(p,x)$ with respect to p respectively.

$$ V'(p,x) = 30(q(p,x) + q'(p,x)p) + (1 - \chi(p,x))\delta T'(p,x)V(x') $$

$$ V''(p,x) = 30(2q'(p,x) + q''(p,x)) + (1 - \chi(p,x))\delta T''(p,x)V(x') - \chi(p,x)\frac{(\delta T'(p,x)V(x'))^2}{\phi(x)} $$

In code:
<code>
while dP>.1:
  P1 = P0 - dV_s(P0,P_old,s_old,V_old,theta,phi_bar,t,params)/d2V_s(P0,P_old,s_old,V_old,theta,phi_bar,t,params)
  P1 = np.where(np.isnan(P1) == True,P_old,np.where((P1<0),0,np.where((P1>1000),1000,P1)))
  dP = np.max(np.abs(P1 - P0))
  P0 = P1
</code>

  ### Value Function Update

Having found $p$ that solves the host's pricing problem, we compute the new value function. 

<code>dV_s(p,P,s,V,theta,\phi_bar,t,params)</code>
