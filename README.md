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
<tr><td style="text-align:left"></td><td>p</td><td>100 * booked</td></tr>
<tr><td style="text-align:left"></td><td>(1)</td><td>(2)</td></tr>
<tr><td colspan="3" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">K</td><td>2.897<sup>***</sup></td><td>3.438<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(0.013)</td><td>(0.071)</td></tr>
<tr><td style="text-align:left"></td><td></td><td></td></tr>
<tr><td style="text-align:left">N</td><td>-0.834<sup>***</sup></td><td>-1.807<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(0.004)</td><td>(0.022)</td></tr>
<tr><td style="text-align:left"></td><td></td><td></td></tr>
<tr><td style="text-align:left">p</td><td></td><td>-0.315<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td></td><td>(0.008)</td></tr>
<tr><td style="text-align:left"></td><td></td><td></td></tr>
<tr><td style="text-align:left">beta.2.</td><td>3.180<sup>***</sup></td><td>4.566<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(0.041)</td><td>(0.213)</td></tr>
<tr><td style="text-align:left"></td><td></td><td></td></tr>
<tr><td style="text-align:left">beta.3.</td><td>5.272<sup>***</sup></td><td>8.048<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(0.041)</td><td>(0.213)</td></tr>
<tr><td style="text-align:left"></td><td></td><td></td></tr>
<tr><td style="text-align:left">beta.4.</td><td>10.070<sup>***</sup></td><td>11.762<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(0.040)</td><td>(0.220)</td></tr>
<tr><td style="text-align:left"></td><td></td><td></td></tr>
<tr><td style="text-align:left">K:N</td><td>-0.049<sup>***</sup></td><td>-0.025<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(0.001)</td><td>(0.003)</td></tr>
<tr><td style="text-align:left"></td><td></td><td></td></tr>
<tr><td style="text-align:left">Constant</td><td>91.948<sup>***</sup></td><td>58.089<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(0.042)</td><td>(0.750)</td></tr>
<tr><td style="text-align:left"></td><td></td><td></td></tr>
<tr><td colspan="3" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">Observations</td><td>433,079</td><td>433,079</td></tr>
<tr><td style="text-align:left">R<sup>2</sup></td><td>0.411</td><td>0.026</td></tr>
<tr><td style="text-align:left">Adjusted R<sup>2</sup></td><td>0.411</td><td>0.026</td></tr>
<tr><td style="text-align:left">Residual Std. Error</td><td>9.147 (df = 433072)</td><td>47.022 (df = 433071)</td></tr>
<tr><td style="text-align:left">F Statistic</td><td>50,345.540<sup>***</sup> (df = 6; 433072)</td><td>1,645.427<sup>***</sup> (df = 7; 433071)</td></tr>
<tr><td colspan="3" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"><em>Note:</em></td><td colspan="2" style="text-align:right"><sup>*</sup>p<0.1; <sup>**</sup>p<0.05; <sup>***</sup>p<0.01</td></tr>
</table>


> stargazer(lm1,lm2,type="html")> > 
> stargazer(lm1,lm2,type="html")

<table style="text-align:center"><tr><td colspan="3" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"></td><td colspan="2"><em>Dependent variable:</em></td></tr>
<tr><td></td><td colspan="2" style="border-bottom: 1px solid black"></td></tr>
<tr><td style="text-align:left"></td><td>(1)</td><td>(2)</td></tr>
<tr><td style="text-align:left"></td><td>p</td><td>$100 * booked$</td></tr>
<tr><td colspan="3" style="border-bottom: 1px solid black">
<tr><td style="text-align:left">Constant</td><td>91.948<sup>***</sup></td><td>58.089<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(0.042)</td><td>(0.750)</td></tr>
<tr><td style="text-align:left"></td><td></td><td></td></tr>
</td></tr><tr><td style="text-align:left">$K$</td><td>2.897<sup>***</sup></td><td>3.438<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(0.013)</td><td>(0.071)</td></tr>
<tr><td style="text-align:left"></td><td></td><td></td></tr>
<tr><td style="text-align:left">$N$</td><td>-0.834<sup>***</sup></td><td>-1.807<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(0.004)</td><td>(0.022)</td></tr>
<tr><td style="text-align:left"></td><td></td><td></td></tr>
<tr><td style="text-align:left">$p$</td><td></td><td>-0.315<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td></td><td>(0.008)</td></tr>
<tr><td style="text-align:left"></td><td></td><td></td></tr>
<tr><td style="text-align:left">mid-scale</td><td>3.180<sup>***</sup></td><td>4.566<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(0.041)</td><td>(0.213)</td></tr>
<tr><td style="text-align:left"></td><td></td><td></td></tr>
<tr><td style="text-align:left">up-scale</td><td>5.272<sup>***</sup></td><td>8.048<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(0.041)</td><td>(0.213)</td></tr>
<tr><td style="text-align:left"></td><td></td><td></td></tr>
<tr><td style="text-align:left">luxury</td><td>10.070<sup>***</sup></td><td>11.762<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(0.040)</td><td>(0.220)</td></tr>
<tr><td style="text-align:left"></td><td></td><td></td></tr>
<tr><td style="text-align:left">$K\times N$</td><td>-0.049<sup>***</sup></td><td>-0.025<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(0.001)</td><td>(0.003)</td></tr>
<tr><td style="text-align:left"></td><td></td><td></td></tr>
<tr><td colspan="3" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">Observations</td><td>433,079</td><td>433,079</td></tr>
<tr><td style="text-align:left">R<sup>2</sup></td><td>0.411</td><td>0.026</td></tr>
<tr><td style="text-align:left">Adjusted R<sup>2</sup></td><td>0.411</td><td>0.026</td></tr>
<tr><td style="text-align:left">Residual Std. Error</td><td>9.147 (df = 433072)</td><td>47.022 (df = 433071)</td></tr>
<tr><td style="text-align:left">F Statistic</td><td>50,345.540<sup>***</sup></td><td>1,645.427<sup>***</sup> </td></tr>
<tr><td colspan="3" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"><em>Note:</em></td><td colspan="2" style="text-align:right"><sup>*</sup>p<0.1; <sup>**</sup>p<0.05; <sup>***</sup>p<0.01</td></tr>
</table>

## Demand estimation

## Supply estimation

## Counterfactual

## Effect decomposition

%### Replication code

%**Replication code**
