# Hidden Markov Models

![H](https://miro.medium.com/max/800/1*0xjHjL19uK0d6llcEJ0Z0w.png)

## Tasks

### [Markov Chain](./0-markov_chain.py)
- Write the function def markov_chain(P, s, t=1): that determines the probability of a markov chain being in a particular state after a specified number of iterations

### [Regular Chains](./1-regular.py)
- Write the function def regular(P): that determines the steady state probabilities of a regular markov chain

### [Absorbing Chains](./2-absorbing.py)
- Write the function def absorbing(P): that determines if a markov chain is absorbing

### [The Forward Algorithm](./3-forward.py)
- Write the function def forward(Observation, Emission, Transition, Initial): that performs the forward algorithm for a hidden markov model

### [The Viretbi Algorithm](./4-viterbi.py)
- Write the function def viterbi(Observation, Emission, Transition, Initial): that calculates the most likely sequence of hidden states for a hidden markov model

### [The Backward Algorithm](./5-backward.py)
- Write the function def backward(Observation, Emission, Transition, Initial): that performs the backward algorithm for a hidden markov model

### [The Baum-Welch Algorithm](./6-baum_welch.py)
- Write the function def baum_welch(Observations, Transition, Emission, Initial, iterations=1000): that performs the Baum-Welch algorithm for a hidden markov model
