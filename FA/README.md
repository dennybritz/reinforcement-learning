## Function Approximation

### Learning Goals

- Understand the motivation for Function Approximation over Table Lookup
- Understand how to incorporate function approximation into existing algorithms
- Understand convergence properties of function approximators and algorithms
- Understand batching using experience replay


### Summary

- Building a big table, one value for each state or state-action pair, is memory- and data-inefficient. Function Approximation can generalize to unseen states using a featurized state representation.
- Treat it as supervised learning problem with the MC- or TD-target as the label and the current state/action as the input. Often the target also depends on the function estimator buy we simply ignore its gradient. So these methods are called semi-gradient methods.
- Challenge: We have non-stationary (policy changes, bootstrapping) and non-iid (correlated in time) data.
- Many methods assume that our action space is discrete because they rely on calculating the argmax over all actions. For large or continuous action spaces are ongoing research.
- For Control, very few guarantees that function approximation converges. For non-linear approximators basically no guarantees at all.
- Experience Replay: Store experience as dataset, randomize it, and repeatedly apply minibatch SGD.
- Tricks to stabilize nonlinear function approximators: Fixed Targets. The target is calculated based on frozen parameter values from a previous time step.
- For the non-episodic (continuing) case function approximation is more complex and we need to give up discounting and use an "average reward" formulation.


### Lectures & Readings

**Required:**

- David Silver's RL Course Lecture 6 - Value Function Approximation ([video](https://www.youtube.com/watch?v=UoPei5o4fps), [slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/FA.pdf))
- [Reinforcement Learning: An Introduction](https://www.dropbox.com/s/d6fyn4a5ag3atzk/bookdraft2016aug.pdf) - Chapter 9: On-policy Prediction with Approximation
- [Reinforcement Learning: An Introduction](https://www.dropbox.com/s/d6fyn4a5ag3atzk/bookdraft2016aug.pdf) - Chapter 10: On-policy Control with Approximation

**Optional:**

- [Tutorial: Introduction to Reinforcement Learning with Function Approximation](https://www.youtube.com/watch?v=ggqnxyjaKe4)


### Exercises

- Solve Mountain Car Problem using Q-Learning with Linear Function Approximation ([Exercise]() [Solution](Q-Learning with Value Function Approximation Solution.ipynb))
- Add Experience Replay to Q-Learning Implementation
