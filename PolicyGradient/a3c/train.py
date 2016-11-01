import unittest
import gym
import sys
import os
import numpy as np
import tensorflow as tf
import itertools
import shutil
import threading
import multiprocessing

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
  sys.path.append(import_path)

# from lib import plotting
from estimators import ValueEstimator, PolicyEstimator
from policy_eval import PolicyEval
from worker import Worker


def make_env():
    return gym.envs.make("Breakout-v0")

VALID_ACTIONS = [0, 1, 2, 3]
T_MAX = 5
NUM_WORKERS = multiprocessing.cpu_count()

# Model Directory
model_dir = "/tmp/a3c"
shutil.rmtree(model_dir, ignore_errors=True)
os.makedirs(model_dir)
summary_writer = tf.train.SummaryWriter(model_dir)

tf.reset_default_graph()
global_step = tf.Variable(0, name="global_step", trainable=False)

with tf.variable_scope("global") as vs:
    value_net = ValueEstimator()
    policy_net = PolicyEstimator(num_outputs=len(VALID_ACTIONS), reuse=True)

global_counter = itertools.count()

workers = []
for worker_id in range(NUM_WORKERS):
    worker = Worker(
        name="worker_{}".format(worker_id),
        env=make_env(),
        policy_net=policy_net,
        value_net=value_net,
        global_counter=global_counter,
        discount_factor = 0.99,
        summary_writer=summary_writer)
    workers.append(worker)

pe = PolicyEval(
    env=make_env(),
    policy_net=policy_net,
    summary_writer=summary_writer)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()

    # Start workers
    worker_threads = []
    for worker in workers:
        worker_fn = lambda: worker.run(sess, coord, T_MAX)
        t = threading.Thread(target=worker_fn)
        t.start()
        worker_threads.append(t)

    # Start a thread for policy eval job
    threading.Thread(target=lambda: pe.continuous_eval(30, sess)).start()

    # Wait for all workers to finish
    coord.join(worker_threads)
