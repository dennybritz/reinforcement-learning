#### Overview and Structure

The goal for this repository is to become a comprehensive tutorial of Reinforcement Learning techniques. The focus is on code examples and exercises. To avoid reinventing the wheel I reference outside learning materials to introduce new concepts. These resources are usually from:

- [David Silver's Reinforcement Learning Course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
- [Reinforcement Learning: An Introduction](https://www.dropbox.com/s/b3psxv2r0ccmf80/book2015oct.pdf)
- [Reinforcement Learning at Georgia Tech (CS 8803)](https://www.udacity.com/course/reinforcement-learning--ud600)
- Various Research papers

All code is written in Python 3 and the RL environments are taken from [OpenAI Gym](https://gym.openai.com/). Advanced techniques use [Tensorflow](tensorflow.org/) for neural network implementations.


#### Contents


- [Introduction to RL problems, OpenAI gym](Introduction/) (08/28/2016)
- [MDPs and Bellman Equations](MDP/) (08/28/2016)
- [Model-Based RL: Policy and Value Iteration using Dynamic Programming](DP/) (08/28/2016)
- [Model-Free Prediction & Control with Monte Carlo (MC)](MC/) (08/28/2016)
- [Model-Free Prediction & Control with Temporal Difference (TD)](TD/) (09/04/2016)
- [Function Approximation](FA/) (09/11/2016)
- [Deep Q Learning](DeepQ/) (09/18/2016)
- [Policy Gradient Methods](PolicyGradient/) (09/25/2016)
- Learning and Planning (TBD)
- Exploration and Exploitation (TBD)


#### References

Classes:

- [David Silver's Reinforcement Learning Course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
- [Reinforcement Learning: An Introduction](https://webdocs.cs.ualberta.ca/~sutton/book/the-book.html)
- [Reinforcement Learning at Georgia Tech (CS 8803)](https://www.udacity.com/course/reinforcement-learning--ud600)
- [Berkeley's CS 294: Deep Reinforcement Learning, Fall 2015](http://rll.berkeley.edu/deeprlcourse/)

Talks:

- [Tutorial: Introduction to Reinforcement Learning with Function Approximation](https://www.youtube.com/watch?v=ggqnxyjaKe4)
- [John Schulman - Deep Reinforcement Learning (4 Lectures)](https://www.youtube.com/playlist?list=PLjKEIQlKCTZYN3CYBlj8r58SbNorobqcp)
- [Deep Reinforcement Learning ICML 2016 Tutorial (David Silver)](http://techtalks.tv/talks/deep-reinforcement-learning/62360/)

Projects:

- [carpedm20/deep-rl-tensorflow](https://github.com/carpedm20/deep-rl-tensorflow)
- [matthiasplappert/keras-rl](https://github.com/matthiasplappert/keras-rl)

Papers:

- [Human-Level Control through Deep Reinforcement Learning (2015-02)](http://www.readcube.com/articles/10.1038/nature14236)
- [Deep Reinforcement Learning with Double Q-learning (2015-09)](http://arxiv.org/abs/1509.06461)
- [Prioritized Experience Replay (2015-11)](http://arxiv.org/abs/1511.05952)
- [Dueling Network Architectures for Deep Reinforcement Learning (2015-11)](http://arxiv.org/abs/1511.06581)
- [Asynchronous Methods for Deep Reinforcement Learning (2016-02)](http://arxiv.org/abs/1602.01783)
- [Deep Reinforcement Learning from Self-Play in Imperfect-Information Games (2016-03)](http://arxiv.org/abs/1603.01121)