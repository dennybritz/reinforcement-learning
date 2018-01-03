## Function Approximation

### Learning Goals

- Understand the motivation for Function Approximation over Table Lookup
- Understand how to incorporate function approximation into existing algorithms
- Understand convergence properties of function approximators and RL algorithms
- Understand batching using experience replay


### Summary

- Building a big table, one value for each state or state-action pair, is memory- and data-inefficient. Function Approximation can generalize to unseen states by  using a featurized state representation.
- Treat RL as supervised learning problem with the MC- or TD-target as the label and the current state/action as the input. Often the target also depends on the function estimator but we simply ignore its gradient. That's why these methods are called semi-gradient methods.
- Challenge: We have non-stationary (policy changes, bootstrapping) and non-iid (correlated in time) data.
- Many methods assume that our action space is discrete because they rely on calculating the argmax over all actions. Large and continuous action spaces are ongoing research.
- For Control very few convergence guarantees exist. For non-linear approximators there are basically no guarantees at all. But they tend to work in practice.
- Experience Replay: Store experience as dataset, randomize it, and repeatedly apply minibatch SGD.
- Tricks to stabilize non-linear function approximators: Fixed Targets. The target is calculated based on frozen parameter values from a previous time step.
- For the non-episodic (continuing) case function approximation is more complex and we need to give up discounting and use an "average reward" formulation.


### Lectures & Readings

**Required:**

- David Silver's RL Course Lecture 6 - Value Function Approximation ([video](https://www.youtube.com/watch?v=UoPei5o4fps), [slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/FA.pdf))
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/bookdraft2018jan1.pdf) - Chapter 9: On-policy Prediction with Approximation
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/bookdraft2018jan1.pdf) - Chapter 10: On-policy Control with Approximation

**Optional:**

- [Tutorial: Introduction to Reinforcement Learning with Function Approximation](https://www.youtube.com/watch?v=ggqnxyjaKe4)


### Exercises

- Get familiar with the [Mountain Car Playground](MountainCar%20Playground.ipynb)

- Solve Mountain Car Problem using Q-Learning with Linear Function Approximation
  - [Exercise](Q-Learning%20with%20Value%20Function%20Approximation.ipynb)
  - [Solution](Q-Learning%20with%20Value%20Function%20Approximation%20Solution.ipynb)
