# Bayesian Probability

![Bay](https://www.norledgemaths.com/uploads/1/0/8/1/10815708/105228283.png)

## Tasks

### [Likelihood](./0-likelihood.py)
- You are conducting a study on a revolutionary cancer drug and are looking to find the probability that a patient who takes this drug will develop severe side effects. During your trials, n patients take the drug and x patients develop severe side effects. You can assume that x follows a binomial distribution

### [Intersection](./1-intersection.py)
- Based on 0-likelihood.py, write a function def intersection(x, n, P, Pr): that calculates the intersection of obtaining this data with the various hypothetical probabilities

### [Marginal Probability](./2-marginal.py)
- Based on 1-intersection.py, write a function def marginal(x, n, P, Pr): that calculates the marginal probability of obtaining the data

### [Posterior](./3-posterior.py)
- Based on 2-marginal.py, write a function def posterior(x, n, P, Pr): that calculates the posterior probability for the various hypothetical probabilities of developing severe side effects given the data
