## Policy Gradient Methods


### Learning Goals

- Understand the difference between value-based and policy-based Reinforcement Learning
- Understand the REINFORCE Algorithm (Monte Carlo Policy Gradient)
- Understand Actor-Critic (AC) algorithms
- Understand Advantage Functions
- Understand Deterministic Policy Gradients (Optional)
- Understand how to scale up Policy Gradient methods using asynchronous actor-critic and Neural Networks (Optional)


### Summary

- Idea: Instead of parameterizing the value function and doing greedy policy improvement we parameterize the policy and do gradient descent into a direction that improves it.
- Sometimes the policy is easier to approximate than the value function. Also, we need a parameterized policy to deal with continuous action spaces and environments where we need to act stochastically.
- Policy Score Function `J(theta)`: Intuitively, it measures how good our policy is. For example, we can use the average value or average reward under a policy as our objective.
- Common choices for the policy function: Softmax for discrete actions, Gaussian parameters for continuous actions.
- Policy Gradient Theorem: `grad(J(theta)) = Ex[grad(log(pi(s, a))) * Q(s, a)]`. Basically, we move our policy into a direction of more reward.
- REINFORCE (Monte Carlo Policy Gradient): We substitute a samples return `g_t` form an episode for Q(s, a) to make an update. Unbiased but high variance.
- Baseline: Instead of measuring the absolute goodness of an action we want to know how much better than "average" it is to take an action given a state. E.g. some states are naturally bad and always give negative reward. This is called the advantage and is defined as `Q(s, a) - V(s)`. We use that for our policy update, e.g. `g_t - V(s)` for REINFORCE.
- Actor-Critic: Instead of waiting until the end of an episode as in REINFORCE we use bootstrapping and make an update at each step. To do that we also train a Critic Q(theta) that approximates the value function. Now we have two function approximators: One of the policy, one for the critic. This is basically TD, but for Policy Gradients.
- A good estimate of the advantage function in the Actor-Critic algorithm is the td error. Our update then becomes `grad(J(theta)) = Ex[grad(log(pi(s, a))) * td_error]`.
- Can use policy gradients with td-lambda, eligibility traces, and so on.
- Deterministic Policy Gradients: Useful for high-dimensional continuous action spaces where stochastic policy gradients are expensive to compute. The idea is to update the policy in the direction of the gradient of the action-value function. To ensure exploration we can use an off-policy actor-critic algorithm with added noise in action selection.
- Deep Deterministic Policy Gradients: Apply tricks from DQN to Deterministic Policy Gradients ;)
- Asynchronous Advantage Actor-Critic (A3C): Instead of using an experience replay buffer as in DQN use multiple agents on different threads to explore the state spaces and make decorrelated updates to the actor and the critic.


### Lectures & Readings

**Required:**

- David Silver's RL Course Lecture 7 - Policy Gradient Methods ([video](https://www.youtube.com/watch?v=KHZVXao4qXs), [slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/pg.pdf))

**Optional:**

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2018.pdf) - Chapter 13: Policy Gradient Methods
- [Deterministic Policy Gradient Algorithms](http://jmlr.org/proceedings/papers/v32/silver14.pdf)
- [Deterministic Policy Gradient Algorithms (Talk)](http://techtalks.tv/talks/deterministic-policy-gradient-algorithms/61098/)
- [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
- [Deep Deterministic Policy Gradients in TensorFlow](http://pemami4911.github.io/blog_posts/2016/08/21/ddpg-rl.html)
- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
- [Deep Reinforcement Learning: A Tutorial (Policy Gradient Section)](http://web.archive.org/web/20161029135055/https://gym.openai.com/docs/rl#id16)



### Exercises

- REINFORCE with Baseline
  - Exercise
  - [Solution](CliffWalk%20REINFORCE%20with%20Baseline%20Solution.ipynb)
- Actor-Critic with Baseline
  - Exercise
  - [Solution](CliffWalk%20Actor%20Critic%20Solution.ipynb)
- Actor-Critic with Baseline for Continuous Action Spaces
  - Exercise
  - [Solution](Continuous%20MountainCar%20Actor%20Critic%20Solution.ipynb)
- Deterministic Policy Gradients for Continuous Action Spaces (WIP)
- Deep Deterministic Policy Gradients (WIP)
- Asynchronous Advantage Actor-Critic (A3C)
  - Exercise
  - [Solution](a3c/)
