# Replication code

## Dataset Preparation

<ul>
  <li>Read in <code>MSA_new-york-newark-jersey-city-ny-nj-pa_Property_Extended_Match_2021-11-19.csv</code> (data on properties) </li>
  <li>Restrict properties to entire homes in Manhattan with exactly 1 bedroom, 1 bathroom, at most 2 guests, at least 1 photo, no pets allowed.</li>
  <li>Read in <code>MSA_new-york-newark-jersey-city-ny-nj-pa_Daily_Match_2021-11-19.csv</code> (daily booking data) for these properties. </li>
  <li>Read in <code>New_York-Newark_Property_Trajectory.csv</code> (~bi-monthly ratings data) and merge with daily bookings data. </li>
  <li>Restrict observation period to 01/01/2016 to 31/12/2019 (4y).</li>
  <li>For each property, drop all observations <l>before</li> first ever booking. AirDNA may have filled in zeros for dates that predate market entry.</li>
  <li>If the number of reviews $N$ is missing for a id-day, fill in the last observed $N$ of the same property.</li>
  <li>If $N$ for a id-day is missing and N has been observe prior, fill in the first observed $N$ of the same property.</li>
  <li>If $N$ is still missing, set $N$ equal to zero.</li>
  <li>Replace zero ratings, with missings and multiply average ratings less or equal to 10 by 10 (coding errors).</li>
  <li>If the rating $r$ is missing for a id-day, fill in the last observed $r$ of the same property.</li>
  <li>If the $r$ is missing for a id-day and no $r$ has been observe prior, fill in the first observed $r$ of the same property.</li>
  <li>If the cleaning fee is missing for a id-day, fill in the last observed cleaning fee of the same property.</li>
  <li>If the cleaning fee for a id-day is missing and cleaning fee has been observe prior, fill in the first observed cleaning fee of the same property.</li>
  <li>If the cleaning fee is still missing or zero, set it to the average across properties.</li>
  <li>Drop all id-days that are anything else but reserved ($B=1$) or available ($B=0$).</li> 
  <li>Drop properties that are never booked during their lifetime.</li>
  <li>Drop properties if their daily rate $p$ is ever at or above the 99 percentile or below the 1 percentile.</li>
  <li>Compute <i>gross</i> daily rates as $$P = p + \frac{\text{cleaning fee}}{\text{avg booking length}}.$$</li>
  <li>Cap $N$ at 20. Define the number of good reviews as $$K = \frac{1}{4}(r-1)N.$$</li>
  <li>Preliminarily define state $x$ as $(N,K)$.</li>
  <li>Load dataset into R and run the regression $ B = a P + \sum b(x) + \sum b(\tau) + \sum b(j) + e$. </li>
  <li>$B = aP$ </li>
</ul>

| type | avg price | avg reviews | avg booking prob |  avg rating |
| ---: | ---: | ---------: | ------: | ------: |
| 1 | \$198.35 | 7.08 | 10.13% | 4.35 stars |
| 2 | \$216,78 | 8.98 | 32.2% | 4.33 stars |
| 3 | \$192.96 | 12.36 | 65.65% | 4.56 stars |
| 4 | \$188.11 | 10.53 | 85.87% | 4.58 stars |

## Demand Estimation


### Model parameters

|  | name |            |  value |
| ---: | ---: | ---------: | ------: |
| demand | price | $\alpha$ | -0.0008 |
|| constant | $\beta_1$ | -14.0368 |
|| mid-scale | $\beta_2$ | -13.7769 |
|| up-scale | $\beta_3$ | -13.7284 |
|| luxury | $\beta_4$ | -13.7971 |
|| quality | $\gamma$ | 5.9013 |
|| prior | $a$ | 9.6511 |
||  | $b$ | 4.664 |
| supply | mean entry cost | $\bar \kappa$ | ? |
|| mean scrap value | $\bar \phi$ | ? |
| other | discount factor | $\delta$ | 0.999 |
|  | revenue fee | $f$ | 0.142 |
|  | arrival rate | $\mu$ | 23,000 |
|  | adjustment prob. | $\upsilon_a$ | 0.019 |
|  | review prob. | $\upsilon_r$ | 0.1817 |
|  | max. no. of reviews | $\bar N$ | 20 |
|  | max. no. of listings | $J$ | 1,800 |

## Monte Carlo simulation

<table style="text-align:center"><tr><td colspan="3" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"></td><td colspan="2"><em>Dependent variable:</em></td></tr>
<tr><td></td><td colspan="2" style="border-bottom: 1px solid black"></td></tr>
<tr><td style="text-align:left"></td><td>(1)</td><td>(2)</td></tr>
<tr><td style="text-align:left"></td><td>$p$</td><td>$100 \times booked$</td></tr>
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
<tr><td style="text-align:left"></td><td></td><td></td></tr>
<tr><td style="text-align:left">$K\times N$</td><td>-0.049<sup>***</sup></td><td>-0.025<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(0.001)</td><td>(0.003)</td></tr>
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
<tr><td colspan="3" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">Observations</td><td>433,079</td><td>433,079</td></tr>
<tr><td style="text-align:left">R<sup>2</sup></td><td>0.411</td><td>0.026</td></tr>
<tr><td style="text-align:left">Adjusted R<sup>2</sup></td><td>0.411</td><td>0.026</td></tr>
<tr><td style="text-align:left">Residual Std. Error</td><td>9.147</td><td>47.022</td></tr>
<tr><td style="text-align:left">F Statistic</td><td>50,345.540<sup>***</sup></td><td>1,645.427<sup>***</sup> </td></tr>
<tr><td colspan="3" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"><em>Note:</em></td><td colspan="2" style="text-align:right"><sup>*</sup>p<0.1; <sup>**</sup>p<0.05; <sup>***</sup>p<0.01</td></tr>
</table>

## Demand estimation

## Supply estimation

## Counterfactual

## Effect decomposition

%### Replication code

%**Replication code**
