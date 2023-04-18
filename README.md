# Replication code

## Model parameters

|  | name |            |  value |
| ---: | ---: | ---------: | ------: |
| demand | price | $\alpha$ | -0.0105 |
|| constant | $\beta_1$ | -13.9974 |
|| mid-scale | $\beta_2$ | 0.1884 |
|| up-scale | $\beta_3$ | 0.3084 |
|| luxury | $\beta_4$ | 0.4544 |
|| quality | $\gamma$ | 3.4554 |
|| prior | $a$ | 7.8516 |
||  | $b$ | 4.3545 |
| supply | mean entry cost | $\bar \kappa$ | 25,106 |
|| mean scrap value | $\bar \phi$ | 18,535 |
| other | discount factor | $\delta$ | 0.99 |
|  | revenue fee | $f$ | 0.142 |
|  | arrival rate | $\mu$ | 20,000 |
|  | adjustment prob. | $\upsilon_r$ | 0.019 |
|  | review prob. | $\upsilon_r$ | 0.2938 $\ln(N_{jt}+2)\frac{4}{7}$ |
|  | max. no. of reviews | $\bar N$ | 20 |
|  | max. no. of listings | $J$ | 2,000 |

## Monte Carlo simulation

|  | (1) |            |  (2) |
|  | $p$ |            |  $100\times booked$ |
| ---: | ---: | ---------: | ------: |
| demand | price | $\alpha$ | -0.0105 |
|| constant | $\beta_1$ | -13.9974 |
|| mid-scale | $\beta_2$ | 0.1884 |
|| up-scale | $\beta_3$ | 0.3084 |

||(1)|(2)|
||$p$|$100\times booked$|
|K|                               2.408***     |                   1.206***    |       
||                                 (0.002)    |                     (0.140)    |                                                                          
|N|                               -1.183***                       -0.776***          
||                                 (0.001)                         (0.064)           
                                                                                   
p                                                               0.627***           
                                                                 (0.051)           
                                                                                   
beta.2.                         3.107***                        2.232***           
                                 (0.006)                         (0.265)           
                                                                                   
beta.3.                         5.249***                        3.082***           
                                 (0.006)                         (0.339)           
                                                                                   
beta.4.                         8.335***                        4.637***           
                                 (0.006)                         (0.471)           
                                                                                   
K:N                             -0.018***                       -0.007**           
                                (0.0001)                         (0.003)           
                                                                                   
Constant                        96.452***                      -32.612***          
                                 (0.006)                         (4.907)           
                                                                                   
-----------------------------------------------------------------------------------
Observations                     432,386                         432,386           
R2                                0.958                           0.024            
Adjusted R2                       0.958                           0.024            
Residual Std. Error        1.407 (df = 432379)            47.025 (df = 432378)     
F Statistic         1,625,097.000*** (df = 6; 432379) 1,529.817*** (df = 7; 432378)
===================================================================================
Note:                                                   *p<0.1; **p<0.05; ***p<0.01

## Demand estimation

## Supply estimation

## Counterfactual

## Effect decomposition

%### Replication code

%**Replication code**
