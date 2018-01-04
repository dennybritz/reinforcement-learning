## Model-Free Prediction & Control with Monte Carlo (MC)


### Learning Goals

- Understand the difference between Prediction and Control
- Know how to use the MC method for predicting state values and state-action values
- Understand the on-policy first-visit MC control algorithm
- Understand off-policy MC control algorithms
- Understand Weighted Importance Sampling
- Understand the benefits of MC algorithms over the Dynamic Programming approach


### Summary

- Dynamic Programming approaches assume complete knowledge of the environment (the MDP). In practice, we often don't have full knowledge of how the world works.
- Monte Carlo (MC) methods can learn directly from experience collected by interacting with the environment. An episode of experience is a series of `(State, Action, Reward, Next State)` tuples.
- MC methods work based on episodes. We sample episodes of experience and make updates to our estimates at the end of each episode. MC methods have high variance (due to lots of random decisions within an episode) but are unbiased.
- MC Policy Evaluation: Given a policy, we want to estimate the state-value function V(s). Sample episodes of experience and estimate V(s) to be the reward received from that state onwards averaged across all of your experience. The same technique works for the action-value function Q(s, a). Given enough samples, this is proven to converge.
- MC Control: Idea is the same as for Dynamic Programming. Use MC Policy Evaluation to evaluate the current policy then improve the policy greedily. The Problem: How do we ensure that we explore all states if we don't know the full environment?
- Solution to exploration problem: Use epsilon-greedy policies instead of full greedy policies. When making a decision act randomly with probability epsilon. This will learn the optimal epsilon-greedy policy.
- Off-Policy Learning: How can we learn about the actual optimal (greedy) policy while following an exploratory (epsilon-greedy) policy? We can use importance sampling, which weighs returns by their probability of occurring under the policy we want to learn about.


### Lectures & Readings

**Required:**

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/bookdraft2018jan1.pdf) - Chapter 5: Monte Carlo Methods


**Optional:**

- David Silver's RL Course Lecture 4 - Model-Free Prediction ([video](https://www.youtube.com/watch?v=PnHCvfgC_ZA), [slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/MC-TD.pdf))
- David Silver's RL Course Lecture 5 - Model-Free Control ([video](https://www.youtube.com/watch?v=0g4j2k_Ggc4), [slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/control.pdf))


### Exercises

- Get familiar with the [Blackjack environment (Blackjack-v0)](Blackjack%20Playground.ipynb)
- Implement the Monte Carlo Prediction to estimate state-action values
  - [Exercise](MC%20Prediction.ipynb)
  - [Solution](MC%20Prediction%20Solution.ipynb)
- Implement the on-policy first-visit Monte Carlo Control algorithm
  - [Exercise](MC%20Control%20with%20Epsilon-Greedy%20Policies.ipynb)
  - [Solution](MC%20Control%20with%20Epsilon-Greedy%20Policies%20Solution.ipynb)
- Implement the off-policy every-visit Monte Carlo Control using Weighted Important Sampling algorithm
  - [Exercise](Off-Policy%20MC%20Control%20with%20Weighted%20Importance%20Sampling.ipynb)
  - [Solution](Off-Policy%20MC%20Control%20with%20Weighted%20Importance%20Sampling%20Solution.ipynb)
