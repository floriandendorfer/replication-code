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
  <li> The moment conditions are $$ \begin{bvector} 1 \\ 2 \\ 3 \end{bvector} $$ </li>
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
  <li> $\alpha$ is most likely biased because it is correlated with unobserved </li>
</ul>


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
