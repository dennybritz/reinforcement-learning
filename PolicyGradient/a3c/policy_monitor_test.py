import gym
import sys
import os
import itertools
import collections
import unittest
import numpy as np
import tensorflow as tf
import tempfile

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
  sys.path.append(import_path)

# from lib import plotting
from lib.atari.state_processor import StateProcessor
from lib.atari import helpers as atari_helpers
from policy_monitor import PolicyMonitor
from estimators import ValueEstimator, PolicyEstimator

def make_env():
  return gym.envs.make("Breakout-v0")

VALID_ACTIONS = [0, 1, 2, 3]

class PolicyMonitorTest(tf.test.TestCase):
  def setUp(self):
    super(PolicyMonitorTest, self).setUp()

    self.env = make_env()
    self.global_step = tf.Variable(0, name="global_step", trainable=False)
    self.summary_writer = tf.train.SummaryWriter(tempfile.mkdtemp())

    with tf.variable_scope("global") as vs:
      self.global_policy_net = PolicyEstimator(len(VALID_ACTIONS))
      self.global_value_net = ValueEstimator(reuse=True)

  def testEvalOnce(self):
    pe = PolicyMonitor(
      env=self.env,
      policy_net=self.global_policy_net,
      summary_writer=self.summary_writer)

    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
      total_reward, episode_length = pe.eval_once(sess)
      self.assertTrue(episode_length > 0)


if __name__ == '__main__':
  unittest.main()