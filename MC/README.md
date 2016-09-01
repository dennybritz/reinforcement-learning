## Model-Free Prediction & Control with Monte Carlo (MC)


### Learning Goals

- Understand the difference between Prediction and Control
- Know how to use the MC method for predicting state values and state-action values
- Understand the on-policy first-visit MC control algorithm
- Understand the off-policy every-visit MC control algorithm
- Understand the benefits of MC algorithms over the DP approach


### Lectures & Readings

**Required:**

- [Reinforcement Learning: An Introduction](https://www.dropbox.com/s/b3psxv2r0ccmf80/book2015oct.pdf) - Chapter 5: Monte Carlo Methods


**Optional:**

- David Silver's RL Course Lecture 4 - Model-Free Prediction ([video](https://www.youtube.com/watch?v=PnHCvfgC_ZA), [slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/MC-TD.pdf))
- David Silver's RL Course Lecture 5 - Model-Free Control ([video](https://www.youtube.com/watch?v=0g4j2k_Ggc4), [slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/control.pdf))


### Exercises

1. [Get familar with the Blackjack environment (Blackjack-v0)](Blackjack Playground.ipynb)

2. [Implement the Monte Carlo Prediction to estimate state-action values in Python](MC Prediction.ipynb) ([Solution]([Implement the Monte Carlo Prediction to estimate state-action values in Python](MC Prediction (Solution).ipynb)))

3. [Implement the on-policy first-visit Monte Carlo Control algorithm in Python](MC Control with Epsilon-Greedy Policies) ([Solution](http://localhost:8890/notebooks/MC/MC%20Control%20with%20Epsilon-Greedy%20Policies%20(Solution).ipynb))

4. [Implement the off-policy every-visit Monte Carlo Control using Weighted Important Sampliing algorithm in Python](Off-Policy MC Control with Weighted Importance Sampling) ([Solution](Off-Policy MC Control with Weighted Importance Sampling (Solution)))