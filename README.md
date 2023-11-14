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

Function <code>U_s(p,theta,t,params)</code> characterizes a guests's indirect **utility** of renting a property in state $x=(N,K,type)$.

  $$u_x = \gamma\frac{a + K(x)}{a + b + N(x)} + \sum_j\beta_j\mathbb{1}(type = j) + \alpha ((1+f)p- t) + \epsilon = V(x) + \epsilon$$
  
$P(x)$ is the rental rate of the listing; $t$ is the counterfactual subsidy. For the moment, we set $t$ equal to zero.

**Observed quality**: Each listing has observed type 1,2,3 or 4.

**Unobserved quality**: The unobserved quality $\omega$ is unknown to guests and hosts. However, $\omega$ is known to be iid $Beta(a,b)$ distributed. After observing the number of good reviews $K$ and bad reviews $N-K$ agents form an expectation about the unobserved quality, $E[\omega|N,K]$.</li>

$\epsilon$ is iid T1EV extreme value distributed.

Function <code>ccp_s(p,P,s,theta,t,params)</code> characterizes the probability that a guest ***intends*** to book the property at rate $p$ provided that all remaining hosts set their prices according to $P(x)$.

$$ccp(p,x) = \frac{\exp(V(p,x))}{1+\sum_xs(x)\exp(V(P(x),x))}$$

**State distribution**: $s(x)$ pins down the number of properties in each state. For later use, we also work out the first-order (<code>dccp_s(p,P,s,theta,t,params)</code>) and second-order (<code>d2ccp_s(p,P,s,theta,t,params)</code>) derivatives of $ccp(p,x)$ with respect to $p$.

$$ccp'(p,x) = ccp(p,x)(1 - ccp(p,x))\alpha(1+f) $$

$$ccp''(p,x) = ccp(p,x)(ccp(p,x)^2 - ccp(p,x))\alpha^2(1+f)^2 $$

**Demand**: The number of arriving guests is $Poisson(\mu)$ distributed. Function <code>q_s(p,P,s,theta,t,params)</code> characterizes the probability that at least one of these consumers books the property, again assuming its rental rate is $p$ while everyone else follows the pricing rule $P(x)$.

$$q(p,x) = 1 - \exp(-\mu \cdot ccp(p,x))$$

Function <code>d2q_s(p,P,s,theta,t,params)</code> and Function <code>d2q_s(p,P,s,theta,t,params)</code> describe the first- and second-order derivatives of $q(p,x)$ with respect to $p$.

  $$q'(p,x) = \mu\exp(-\mu \cdot ccp(p,x))ccp'(p,x)$$
  $$q''(p,x) = \mu\exp(-\mu \cdot ccp(p,x))(ccp''(p,x)-\mu\cdot ccp'(p,x))$$

## State Transitions

If a property is booked ($q(p,x) = 1$), $x$ changes with probability $\upsilon_r = 70.41\%$. Conditional on being booked, it receives a good review ($\Delta N = 1, \Delta K = 1$) with probability $\frac{a+K(x)}{a+b+N(x)}$. Conditional on being booked, it receives a bad review ($\Delta N = 1, \Delta K = 0$) with probability $\left(1-\frac{a + K(x)}{a+b+N(x)}\right)$. The **probability of getting a good review**  and the **probability of getting a bad review** are $\rho^g(p,x)$ and $\rho^b(p,x)$ respectively. States where $N=20$ are **terminal** and the probability of getting a review is zero.

$$\rho^g(p,x) = \upsilon_rq(p,x)\frac{a+K(x)}{a+b+N(x)}$$
$$\rho^b(p,x) = \upsilon_rq(p,x)\left(1-\frac{a + K(x)}{a+b+N(x)}\right)$$

Accordingly, the probability $\rho^0(p,x)$ of getting no review is $1-\rho^g(p,x)-\rho^b(p,x)$. States are arranged in increasing order of $type$ and, for a given type, in increasing order of $N$ and, for a given $N$, in increasing order of $K$. $S$ is the **state space**.

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

Function  <code>T_s(q,theta,t,params)</code> stores the **transition matrix** $T(q)$. It turns out that the way states are ordered the number of zeros between $\rho^0(p,x)$ and $\rho^g(p,x)$ is $N$.  

|  | $(0,0,1)$ | $(0,1,1)$ | $(1,1,1)$ | $(0,2,1)$ | $(1,2,1)$ | $(2,2,1)$ | ... | $(20,20,4)$ | 
| :---: | :---: | :---------: | :------: | :------: | :------: | :------: | :------: | :------: |
| $(0,0,1)$ | $\rho^0_{(0,0,1)}$ | $\rho^g_{(0,0,1)}$ | $\rho^b_{(0,0,1)}$ | 0 | 0 | 0 | ... | 0 |
| $(0,1,1)$ | 0 | $\rho^0_{(1,1,1)}$ | 0 | $\rho^g_{(1,1,1)}$ | $\rho^b_{(1,1,1)}$ | 0 | ... | 0 |
| $(1,1,1)$ | 0 | 0 | $\rho^0_{(2,0,1)}$ | 0 | $\rho^g_{(2,0,1)}$ | $\rho^b_{(2,0,1)}$ | ... | 0 |
| $(0,2,1)$ | 0 | 0 | 0 | $\rho^0_{(2,1,1)}$ | 0 | 0 | ... | 0 |
| $(1,2,1)$ | 0 | 0 | 0 | 0 | $\rho^0_{(2,2,1)}$ | 0 | ... | 0 |
| $(2,2,1)$ | 0 | 0 | 0 | 0 | 0 | $\rho^0_{(3,0,1)}$ | ... | 0 |
| ... | ... | ... | ... | ... | ... | ... | ... | 0 |
| $(20,20,1)$ | 0 | 0 | 0 | 0 | 0 | 0 | ... | 1 |

## Market Entry & Exit

