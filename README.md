#### Overview and Structure

The goal for this repository is to become a comprehensive tutorial of Reinforcement Learning techniques. The focus is on practical applications and code examples. This does not mean that theory is completely ignored, just that there will be fewer formal proofs and more code examples than you may find in a typical university course.

Whenever possible, this tutorial references outside learning materials to introduce new concepts. These resources are usually from:

- [David Silver's Reinforcement Learning Course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
- [Reinforcement Learning: An Introduction](https://www.dropbox.com/s/b3psxv2r0ccmf80/book2015oct.pdf)
- [Reinforcement Learning at Georgia Tech (CS 8803)](https://www.udacity.com/course/reinforcement-learning--ud600)
- Various Research papers

All code is written in Python 3 and the RL environments are taken from [OpenAI Gym](https://gym.openai.com/). Advanced techniques use [Tensorflow](tensorflow.org/) for neural network implementations.


#### Contents

- [Introduction to RL problems, OpenAI gym](Introduction/)
- [MDPs and Bellman Equations](MDP/)
- [Model-Based RL: Policy and Value Iteration using Dynamic Programming](DP/)
- [Model-Free Prediction & Control with Monte Carlo (MC)](MC/)
- [Model-Free Prediction & Control with Temporal Difference (TD)](TD/)
- Function Approximation
- Deep Q Learning
- Policy Gradient Methods
- Policy Gradient Methods with Function Approximation
- Asynchronous Policy Gradient Methods (A3C)
- Learning and Planning

#### References

Classes:

- [David Silver's Reinforcement Learning Course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
- [Reinforcement Learning: An Introduction](https://webdocs.cs.ualberta.ca/~sutton/book/the-book.html)
- [Reinforcement Learning at Georgia Tech (CS 8803)](https://www.udacity.com/course/reinforcement-learning--ud600)

Projects:

- [carpedm20/deep-rl-tensorflow](https://github.com/carpedm20/deep-rl-tensorflow)
- [matthiasplappert/keras-rl](https://github.com/matthiasplappert/keras-rl)

Papers

- [Human-Level Control through Deep Reinforcement Learning (2015-02)](http://www.readcube.com/articles/10.1038/nature14236)
- [Deep Reinforcement Learning with Double Q-learning (2015-09)](http://arxiv.org/abs/1509.06461)
- [Prioritized Experience Replay (2015-11)](http://arxiv.org/abs/1511.05952)
- [Dueling Network Architectures for Deep Reinforcement Learning (2015-11)](http://arxiv.org/abs/1511.06581)
- [Asynchronous Methods for Deep Reinforcement Learning (2016-02)](http://arxiv.org/abs/1602.01783)
- [Deep Reinforcement Learning from Self-Play in Imperfect-Information Games (2016-03)](http://arxiv.org/abs/1603.01121)