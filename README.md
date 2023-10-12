# Replication code

## Dataset Preparation

<ul>
  <li>Read in <code>MSA_new-york-newark-jersey-city-ny-nj-pa_Property_Extended_Match_2021-11-19.csv</code> (data on properties) </li>
  <li>Restrict properties to entire homes in Manhattan with exactly 1 bedroom, 1 bathroom, at most 2 guests, at least 1 photo, no pets allowed.</li>
  <li>Read in <code>MSA_new-york-newark-jersey-city-ny-nj-pa_Daily_Match_2021-11-19.csv</code> (daily booking data) for these properties. </li>
  <li>Read in <code>New_York-Newark_Property_Trajectory.csv</code> (~bi-monthly ratings data) and merge with daily bookings data. </li>
  <li>Restrict observation period to 01/01/2016 to 31/12/2019 (4y).</li>
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
  <li>Load dataset into R and run the regression $B = aP + \sum b(x) + \sum b(\tau) + \sum b(j) + e$, where $b(x)$, $b(\tau)$ and $b(j)$ are state, week-year and id dummies respectively </li>
  <li> Types $t\in\{1,2,3,4\}$ are defined based on which quartile $b(j)$ falls into. </li>
</ul>

| type | avg price | avg reviews | avg booking prob | avg rating |
| ---: | ---: | ---------: | ------: | ------: |
| 1 | \$198.97 | 8.09 | 11.22% | 4.33 stars |
| 2 | \$218.30 | 7.60 | 29.48% | 4.32 stars |
| 3 | \$193.54 | 11.71 | 64.14% | 4.57 stars |
| 4 | \$188.83 | 9.65 | 83.56% | 4.57 stars |

## Demand Estimation

<ul>
  <li> On average, there are 635.03 bookings per day. Assuming there are 20,000 consumers every day, Airbnb's market share is 3.17%.  </li>
  <li> Defining a 'month' as a 4 week interval, aggregate the daily data to the month level. </li>
  <li> A property's market share is $s = \frac{B}{mu}$. </li>
  <li> The share of the outside good is $s_0 = 1- \sum s$. </li>
  <li> A property's market share <i>within</i> a certain type is $s_{t} = \frac{B}{\sum_{t}B_t}$. </li>
  <li> Drop month-ids if the market share is zero. 63.04% of the original dataset remain. </li>
  <li> Use GMM to estimate $$\ln(s) - \ln(s_0) = \gamma\frac{\text{expit}(\psi)\exp(\iota) + K}{\exp(\iota) + N} + 1.146\alpha P + \sum b(t) + \xi. $$ </li>
  <li> The moment conditions are $ \xiZ $, where $Z$ includes $N,K,KN,P,t_1,t_2,t_3,t_4$. </li>
</ul>

| coef | estimate | std err | sign |
| ---: | ---: | ---------: | ------: |
| $\psi$ | -0.263 | (0.1630) |  |
| $\iota$ | 1.372 | (0.1311) | *** |
| $\alpha$ | -0.001 | (0.0000) | *** |
| $\beta_1$ | -11.170 | (0.0520) | *** |
| $\beta_2$ | -10.860 | (0.0516) | *** |
| $\beta_3$ | -10.579 | (0.0518) | *** |
| $\beta_4$ | -10.333 | (0.0044) | *** |
| $\gamma$ | 0.764 | (0.0610) | *** |

<ul>
  <li> $\alpha$ is most likely biased because it is correlated with the unobserved quality of a property. </li>
  <li> To address this, define instrument $z$ as the <i>average booking length</i> in days; whether a property tends to be booked short- or long-term should not have a direct effect on the <i>probability</i> that it is booked. However, the marginal cost of providing the Airbnb presumably decreases in the number of days the guest stays at the property (notably the host has to be physically present on the first day). Thus, the reservation length reflects the marginal cost (and, as the marginal cost is partially passed on to guests, is correlated with the rental rate) but does not relate to a property's unobserved quality. </li>
</ul>

| count | 112595 |
| $\iota$ | 1.372 |
| $\alpha$ | -0.001 |
| $\beta_1$ | -11.170 |
| $\beta_2$ | -10.860 |
| $\beta_3$ | -10.579 |
| $\beta_4$ | -10.333 |
| $\gamma$ | 0.764 |

| coef | estimate |
| ---: | ---: |
| count | 112595 |
| mean | 6.9125 |
| std | 5.9358 |
| min | 1.0000 |
| 25% | 4.3333 |
| 50% | 5.7321 |
| 75% | 7.8519 |
| max | 274.0000 |

<ul>
  <li> Test the relevance of $z$ by regressing $p$ on $z$, type dummies, $N, K, NK, N^2, K^2, N^2K, NK^2$ and $N^2K^2$. We find that <i>ceteris paribus</i> a one day longer average reservation length coincides with a 0.023
</ul>

<table style="text-align:center"><tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"></td><td><em>Dependent variable:</em></td></tr>
<tr><td></td><td colspan="1" style="border-bottom: 1px solid black"></td></tr>
<tr><td style="text-align:left"></td><td>p</td></tr>
<tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">z</td><td>-0.023<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(0.005)</td></tr>
<tr><td style="text-align:left"></td><td></td></tr>
<tr><td style="text-align:left">type.1</td><td>288.723<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(10.367)</td></tr>
<tr><td style="text-align:left"></td><td></td></tr>
<tr><td style="text-align:left">type.2</td><td>305.906<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(10.357)</td></tr>
<tr><td style="text-align:left"></td><td></td></tr>
<tr><td style="text-align:left">type.3</td><td>282.740<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(10.361)</td></tr>
<tr><td style="text-align:left"></td><td></td></tr>
<tr><td style="text-align:left">type.4</td><td>277.917<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(10.362)</td></tr>
<tr><td style="text-align:left"></td><td></td></tr>
<tr><td style="text-align:left">poly(N, 2)1</td><td>-36,044.900<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(2,363.653)</td></tr>
<tr><td style="text-align:left"></td><td></td></tr>
<tr><td style="text-align:left">poly(N, 2)2</td><td>7,619.265<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(857.593)</td></tr>
<tr><td style="text-align:left"></td><td></td></tr>
<tr><td style="text-align:left">poly(K, 2)1</td><td>47,937.280<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(3,519.021)</td></tr>
<tr><td style="text-align:left"></td><td></td></tr>
<tr><td style="text-align:left">poly(K, 2)2</td><td>10,007.650<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(1,238.603)</td></tr>
<tr><td style="text-align:left"></td><td></td></tr>
<tr><td style="text-align:left">poly(N, 2)1:poly(K, 2)1</td><td>-9,786,715.000<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(1,117,009.000)</td></tr>
<tr><td style="text-align:left"></td><td></td></tr>
<tr><td style="text-align:left">poly(N, 2)2:poly(K, 2)1</td><td>812,320.900<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(277,768.700)</td></tr>
<tr><td style="text-align:left"></td><td></td></tr>
<tr><td style="text-align:left">poly(N, 2)1:poly(K, 2)2</td><td>-759,138.900<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(277,942.400)</td></tr>
<tr><td style="text-align:left"></td><td></td></tr>
<tr><td style="text-align:left">poly(N, 2)2:poly(K, 2)2</td><td>205,383.500<sup>***</sup></td></tr>
<tr><td style="text-align:left"></td><td>(38,389.670)</td></tr>
<tr><td style="text-align:left"></td><td></td></tr>
<tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">Observations</td><td>105,926</td></tr>
<tr><td style="text-align:left">R<sup>2</sup></td><td>0.907</td></tr>
<tr><td style="text-align:left">Adjusted R<sup>2</sup></td><td>0.907</td></tr>
<tr><td style="text-align:left">Residual Std. Error</td><td>64.011 (df = 105913)</td></tr>
<tr><td style="text-align:left">F Statistic</td><td>79,303.910<sup>***</sup> (df = 13; 105913)</td></tr>
<tr><td colspan="2" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"><em>Note:</em></td><td style="text-align:right"><sup>*</sup>p<0.1; <sup>**</sup>p<0.05; <sup>***</sup>p<0.01</td></tr>
</table>

## Estimation Results

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
