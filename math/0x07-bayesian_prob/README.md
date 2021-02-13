# Bayesian Probability

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/8/8358e1144bbb1fcc51b4.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210213%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210213T021457Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=c5728d71721e71e31fb051bfaf03c664e9175306a612cfef86dc95c75c661303)

## Tasks

### [Likelihood](./0-likelihood.py)
- You are conducting a study on a revolutionary cancer drug and are looking to find the probability that a patient who takes this drug will develop severe side effects. During your trials, n patients take the drug and x patients develop severe side effects. You can assume that x follows a binomial distribution

### [Intersection](./1-intersection.py)
- Based on 0-likelihood.py, write a function def intersection(x, n, P, Pr): that calculates the intersection of obtaining this data with the various hypothetical probabilities

### [Marginal Probability](./2-marginal.py)
- Based on 1-intersection.py, write a function def marginal(x, n, P, Pr): that calculates the marginal probability of obtaining the data

### [Posterior](./3-posterior.py)
- Based on 2-marginal.py, write a function def posterior(x, n, P, Pr): that calculates the posterior probability for the various hypothetical probabilities of developing severe side effects given the data
