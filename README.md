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

<table style="text-align:center"><tr><td colspan="3" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"></td><td colspan="2"><em>Dependent variable:</em></td></tr>
<tr><td></td><td colspan="2" style="border-bottom: 1px solid black"></td></tr>
<tr><td style="text-align:left"></td><td>$p$</td><td>100 * booked</td></tr>
<tr><td style="text-align:left"></td><td>(1)</td><td>(2)</td></tr>
<tr><td colspan="3" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">K</td><td>2.408<sup>***</sup></td><td>1.206<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(0.002)</td><td>(0.140)</td></tr>
<tr><td style="text-align:left"></td><td></td><td></td></tr>
<tr><td style="text-align:left">N</td><td>-1.183<sup>***</sup></td><td>-0.776<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(0.001)</td><td>(0.064)</td></tr>
<tr><td style="text-align:left"></td><td></td><td></td></tr>
<tr><td style="text-align:left">p</td><td></td><td>0.627<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td></td><td>(0.051)</td></tr>
<tr><td style="text-align:left"></td><td></td><td></td></tr>
<tr><td style="text-align:left">beta.2.</td><td>3.107<sup>***</sup></td><td>2.232<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(0.006)</td><td>(0.265)</td></tr>
<tr><td style="text-align:left"></td><td></td><td></td></tr>
<tr><td style="text-align:left">beta.3.</td><td>5.249<sup>***</sup></td><td>3.082<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(0.006)</td><td>(0.339)</td></tr>
<tr><td style="text-align:left"></td><td></td><td></td></tr>
<tr><td style="text-align:left">beta.4.</td><td>8.335<sup>***</sup></td><td>4.637<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(0.006)</td><td>(0.471)</td></tr>
<tr><td style="text-align:left"></td><td></td><td></td></tr>
<tr><td style="text-align:left">K:N</td><td>-0.018<sup>***</sup></td><td>-0.007<sup>**</sup></td></tr>
<tr><td style="text-align:left"></td><td>(0.0001)</td><td>(0.003)</td></tr>
<tr><td style="text-align:left"></td><td></td><td></td></tr>
<tr><td style="text-align:left">Constant</td><td>96.452<sup>***</sup></td><td>-32.612<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(0.006)</td><td>(4.907)</td></tr>
<tr><td style="text-align:left"></td><td></td><td></td></tr>
<tr><td colspan="3" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">Observations</td><td>432,386</td><td>432,386</td></tr>
<tr><td style="text-align:left">R<sup>2</sup></td><td>0.958</td><td>0.024</td></tr>
<tr><td style="text-align:left">Adjusted R<sup>2</sup></td><td>0.958</td><td>0.024</td></tr>
<tr><td style="text-align:left">Residual Std. Error</td><td>1.407 (df = 432379)</td><td>47.025 (df = 432378)</td></tr>
<tr><td style="text-align:left">F Statistic</td><td>1,625,097.000<sup>***</sup> (df = 6; 432379)</td><td>1,529.817<sup>***</sup> (df = 7; 432378)</td></tr>
<tr><td colspan="3" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"><em>Note:</em></td><td colspan="2" style="text-align:right"><sup>*</sup>p<0.1; <sup>**</sup>p<0.05; <sup>***</sup>p<0.01</td></tr>
</table>

## Demand estimation

## Supply estimation

## Counterfactual

## Effect decomposition

%### Replication code

%**Replication code**
