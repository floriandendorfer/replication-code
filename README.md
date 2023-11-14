# Replication code

## Parameter

<center>
|  | name |            |  value |
| ---: | ---: | ---------: | ------: |
| demand | price | $\alpha$ | -0.0105 |
|| type 1 | $\beta_1$ | -11.1007 |
|| type 2 | $\beta_2$ | -10.7004 |
|| type 3 | $\beta_3$ | -10.5486 |
|| type 4 | $\beta_4$ | -10.3437 |
|| quality | $\gamma$ | 3.79019 |
|| prior | $a$ | 7.9686 |
||  | $b$ | 1.4726 |
| supply | mean entry cost (type 1) | $\bar \kappa_1$ | 37,130 |
|  | mean entry cost (type 2) | $\bar \kappa_2$ | 237,883 |
|  | mean entry cost (type 3) | $\bar \kappa_3$ | 182,152 |
|  | mean entry cost (type 4) | $\bar \kappa_4$ | 356,233 |
|| mean fixed cost (type 1) | $\bar \phi_1$ | 2,468 |
|| mean fixed cost (type 2) | $\bar \phi_2$ | 3,037 |
|| mean fixed cost (type 3) | $\bar \phi_3$ | 3,367 |
|| mean fixed cost (type 4) | $\bar \phi_4$ | 3,839 |
| other | discount factor | $\delta$ | 0.995 |
|  | revenue fee | $f$ | 0.142 |
|  | review prob. | $\upsilon_r$ | 0.7041 |
|  | max. no. of reviews | $\bar N$ | 20 |
|  | arrival rate | $\mu$ | 10,000 |
|  | max. no. of listings | $J$ | 10,000 |
</center>

## Demand

<ul>
  <li> Function <code>U_s(p,theta,t,params)</code> characterizes a guests's indirect utility of renting a property in state $x=(N,K,type)$. </li>
  <p style="text-align: center;">$u_{x} = \gamma\frac{a + K(x)}{a + b + N(x)} + \sum_j\beta(x)\mathbb{1}(type = j) + \alpha ((1+f)P(x)- t) + \epsilon$</p>
  <ul>
    <li> $P(x)$ is the rental rate of the listing; $t$ is the counterfactual subsidy. For the moment, we set $t$ equal to zero. </li>
    <li> __Observed quality__: Each listing has observed type 1,2,3 or 4. </li>
    <li> **Unobserved quality**: The unobserved quality $\omega$ is unknown to guests and hosts. However, $\omega$ is known to be iid $Beta(a,b)$ distributed. After observing the number of good reviews $K$ and bad reviews $N-K$ agents form an expectation about the unobserved quality, $E[\omega|N,K]$.</li>
    <li> $\epsilon$ is iid T1EV extreme value distributed.</li>
  </ul>
  <li> Function <code>ccp_s(p,P,s,theta,t,params)</code> characterizes the probability that a listing is booked at rate $p$ provided that all remaining hosts set their prices according to $P(x)$. **State distribution** $s(x)$ pins down the number of properties in each state.   
</ul>
