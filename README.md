# Replication code

## Parameter

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

## Demand

<ul>
  <li> Function <code>U_s(p,theta,t,params)</code> characterizes a consumer's indirect utility of renting a property in state $s$ </li>
  $$ u_{s} = \gamma\frac{a + K(s)}{a + b + N(s)} + \beta(s) + \alpha ((1+f)P(s)- t) + \epsilon $$
  <li> $P(s)$ is the rental rate of the listing; $t$ is the counterfactual subsidy. For the moment, we set $t$ equal to zero. </li>
  <li> **Observed quality**: Listings . </li>
</ul>
