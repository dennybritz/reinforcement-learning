### Overview and Structure

The goal of this repository is to provide code, exercises and solutions of popular Reinforcement Learning algorithms. These are meant to serve as a learning tool to complement the theoretical materials from

- [Reinforcement Learning: An Introduction (2nd Edition)](https://www.dropbox.com/s/d6fyn4a5ag3atzk/bookdraft2016aug.pdf?dl=0)
- [David Silver's Reinforcement Learning Course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)

Each folder in this repository corresponds to one or more chapters of the above textbook and/or course. In addition to exercises and solution each folder also contains a list of learning goals, a brief concept summary, and links to the relevant readings.

All code is written in Python 3 and the RL environments are taken from [OpenAI Gym](https://gym.openai.com/). Advanced techniques use [Tensorflow](tensorflow.org/) for neural network implementations.


### Table of Contents

- [Introduction to RL problems, OpenAI gym](Introduction/)
- [MDPs and Bellman Equations](MDP/)
- [Model-Based RL: Policy and Value Iteration using Dynamic Programming](DP/)
- [Model-Free Prediction & Control with Monte Carlo (MC)](MC/)
- [Model-Free Prediction & Control with Temporal Difference (TD)](TD/)
- [Function Approximation](FA/) (WIP)
- [Deep Q Learning](DeepQ/) (WIP)
- [Policy Gradient Methods](PolicyGradient/) (WIP)
- Learning and Planning (WIP)
- Exploration and Exploitation (WIP)


### Resources

Textbooks:

- [Reinforcement Learning: An Introduction (2nd Edition)](https://www.dropbox.com/s/d6fyn4a5ag3atzk/bookdraft2016aug.pdf)

Classes:

- [David Silver's Reinforcement Learning Course (UCL, 2015)](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
- [CS 8803 - Reinforcement Learning (Georgia Tech)](https://www.udacity.com/course/reinforcement-learning--ud600)
- [CS294 - Deep Reinforcement Learning (Berkeley, Fall 2015)](http://rll.berkeley.edu/deeprlcourse/)

Talks/Tutorials:

- [Introduction to Reinforcement Learning (Joelle Pineau @ Deep Learning Summer School 2016)](http://videolectures.net/deeplearning2016_pineau_reinforcement_learning/)
- [Deep Reinforcement Learning (Pieter Abbeel @ Deep Learning Summer School 2016)](http://videolectures.net/deeplearning2016_abbeel_deep_reinforcement/)
- [Deep Reinforcement Learning ICML 2016 Tutorial (David Silver)](http://techtalks.tv/talks/deep-reinforcement-learning/62360/)
- [Tutorial: Introduction to Reinforcement Learning with Function Approximation](https://www.youtube.com/watch?v=ggqnxyjaKe4)
- [John Schulman - Deep Reinforcement Learning (4 Lectures)](https://www.youtube.com/playlist?list=PLjKEIQlKCTZYN3CYBlj8r58SbNorobqcp)

Other Projects:

- [carpedm20/deep-rl-tensorflow](https://github.com/carpedm20/deep-rl-tensorflow)
- [matthiasplappert/keras-rl](https://github.com/matthiasplappert/keras-rl)

Selected Papers:

- [Human-Level Control through Deep Reinforcement Learning (2015-02)](http://www.readcube.com/articles/10.1038/nature14236)
- [Deep Reinforcement Learning with Double Q-learning (2015-09)](http://arxiv.org/abs/1509.06461)
- [Prioritized Experience Replay (2015-11)](http://arxiv.org/abs/1511.05952)
- [Dueling Network Architectures for Deep Reinforcement Learning (2015-11)](http://arxiv.org/abs/1511.06581)
- [Asynchronous Methods for Deep Reinforcement Learning (2016-02)](http://arxiv.org/abs/1602.01783)
- [Deep Reinforcement Learning from Self-Play in Imperfect-Information Games (2016-03)](http://arxiv.org/abs/1603.01121)