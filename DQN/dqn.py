import gym
import itertools
import matplotlib
import numpy as np
import os
import sys
import shutil
import sklearn.pipeline
import sklearn.preprocessing
import tensorflow as tf

if "../" not in sys.path:
  sys.path.append("../") 

from collections import deque, namedtuple
from lib import plotting

env = gym.envs.make("Breakout-v0")

# Actions: 0 (noop), 1 (fire), 2 (left) and 3 (right) are valid actions
VALID_ACTIONS = [0, 1, 2, 3]

class Estimator():
    """Q-Value Estimator neural network.

    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, scope="estimator", summaries_dir=None):
        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.train.SummaryWriter(summary_dir)

    def preprocess_state(self, s):
        """
        Crop the Atari image to a square.
        This isn't striclty necessary, but it's what's done in the paper
        """
        return s[:, :, 34:-16,:,:]

    def _build_model(self):
        """
        Builds the Tensorflow graph.
        """
        # Placeholders for our input
        # Our input are 4 RGB frames of shape 160, 160 each
        self.X_pl = tf.placeholder(shape=[None, 4, 160, 160, 3], dtype=tf.float32, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        batch_size = tf.shape(self.X_pl)[0]

        # Resize inputs and convert to grayscale
        X_stacked =  tf.reshape(self.X_pl, [-1, 160, 160, 3])
        X_resized = tf.image.resize_images(X_stacked, 84, 84)
        self.X_grayscale = tf.to_float(tf.image.rgb_to_grayscale(X_resized)) / 255.0
        X_stacked = tf.reshape(self.X_grayscale, [batch_size, 4, 84, 84])
        X_stacked = tf.transpose(X_stacked, [0, 2, 3, 1])

        # Three convolutional layers
        conv1 = tf.contrib.layers.conv2d(
            X_stacked, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(
            conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(
            conv2, 64, 3, 1, activation_fn=tf.nn.relu)

        # Fully connected layers
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        self.predictions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS))

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calcualte the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        # Summaries for Tensorboard
        self.summaries = tf.merge_summary([
            tf.scalar_summary("loss", self.loss),
            tf.histogram_summary("loss_hist", self.losses),
            tf.histogram_summary("q_values_hist", self.predictions),
            tf.scalar_summary("max_q_value", tf.reduce_max(self.predictions))
        ])


    def predict(self, sess, s):
        """
        Predicts action values.

        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size, 4, 160, 160, 3]

        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated 
          action values.
        """
        state = self.preprocess_state(s)
        return sess.run(self.predictions, { self.X_pl: state })

    def update(self, sess, s, a, y):
        """
        Updates the estimator towards the given targets.

        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, 4, 160, 160, 3]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        state = self.preprocess_state(s)
        feed_dict = { self.X_pl: state, self.y_pl: y, self.actions_pl: a }
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss


    def predict(self, s):
        sess = tf.get_default_session()
        state = self.preprocess_state(s)
        feed_dict = { self.X_pl: state }
        return sess.run(self.predictions, feed_dict)

    def update(self, s, a, y):
        sess = tf.get_default_session()
        state = self.preprocess_state(s)
        feed_dict = { self.X_pl: state, self.y_pl: y, self.actions_pl: a }
        summaries, global_step, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss


def copy_model_parameters(estimator1, estimator2):
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess = tf.get_default_session()
    sess.run(update_ops)



def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


# In[272]:

def deep_q_learning(sess,
                    env,
                    q_estimator,
                    target_estimator,
                    experiment_dir,
                    num_episodes,
                    replay_memory_size=100000,
                    replay_memory_init_size=50000,
                    update_target_estimator_every=10000,
                    discount_factor=0.99,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500000,
                    batch_size=32):
    """
    Q-Learning algorithm for fff-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        discount_factor: Lambda time discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    replay_memory = deque(maxlen=replay_memory_size)

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    monitor_path = os.path.join(experiment_dir, "monitor")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)

    saver = tf.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(tf.get_default_session(), latest_checkpoint)

    # Populate the replay memory with some random experience
    print("Populating replay memory...\n")
    state = env.reset()
    state = np.array([state] * 4)
    for i in range(replay_memory_init_size):
        action = np.random.choice(len(VALID_ACTIONS))
        next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
        next_state = np.append([next_state], state[1:,:,:,:], axis=0)
        replay_memory.append(Transition(state, action, reward, next_state, done))
        if done:
            state = env.reset()
            state = np.array([state] * 4)
        else:
            state = next_state

    # env.monitor.start(monitor_path, resume=True)
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    for i_episode in range(num_episodes):

        # Save the current checkpoint
        saver.save(tf.get_default_session(), checkpoint_path)

        # Reset the environment and pick the first action
        state = env.reset()
        state = np.array([state] * 4)
        loss = None

        # One step in the environment
        for t in itertools.count():
            total_t = sess.run(tf.contrib.framework.get_global_step())

            epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]

            # The policy we're following
            policy = make_epsilon_greedy_policy(
                q_estimator,
                epsilons[min(total_t, epsilon_decay_steps-1)],
                len(VALID_ACTIONS))

            # Add epsilon to Tensorboard
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=epsilon, tag="epsilon")
            q_estimator.summary_writer.add_summary(episode_summary, total_t)

            if total_t % update_target_estimator_every == 0:
                copy_model_parameters(q_estimator, target_estimator)
                print("\nCopied model parameters to target network.")

            # env.render()

            # Print out which episode we're on, useful for debugging.
            print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(t, total_t, i_episode + 1, num_episodes, loss), end="")
            sys.stdout.flush()

            # Take a step
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
            next_state = np.append([next_state], state[1:,:,:,:], axis=0)

            # Save transition in replay memory
            replay_memory.append(Transition(state, action, reward, next_state, done))

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # Sample from the replay memory
            sample_len = min(batch_size, len(replay_memory))
            sample_idx = np.random.choice(len(replay_memory), sample_len, replace=False)
            samples = [replay_memory[_] for _ in sample_idx]
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

            # Calculate q values and targets
            q_values_next = target_estimator.predict(next_states_batch)
            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount_factor * np.amax(q_values_next, axis=1)

            # Perform gradient descent update
            states_batch = np.array(states_batch)
            loss = q_estimator.update(states_batch, action_batch, targets_batch)

            if done:
                break

            state = next_state


        # Add summaries to tensorboard
        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=stats.episode_rewards[i_episode], node_name="episode_reward", tag="episode_reward")
        episode_summary.value.add(simple_value=stats.episode_lengths[i_episode], node_name="episode_length", tag="episode_length")
        q_estimator.summary_writer.add_summary(episode_summary, total_t)
        q_estimator.summary_writer.flush()

        yield total_t, plotting.EpisodeStats(
            episode_lengths=stats.episode_lengths[:i_episode+1],
            episode_rewards=stats.episode_rewards[:i_episode+1])

    env.monitor.close()
    return stats


tf.reset_default_graph()

# Where we save our checkpoints and graphs
experiment_dir = os.path.abspath("./experiments/{}".format(env.spec.id))

# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)

# Create estimators
q_estimator = Estimator(scope="q", summaries_dir=experiment_dir)
target_estimator = Estimator(scope="target_q")

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for t, stats in deep_q_learning(sess,
                                    env,
                                    q_estimator=q_estimator,
                                    target_estimator=target_estimator,
                                    num_episodes=50000,
                                    replay_memory_init_size=50000,
                                    update_target_estimator_every=10000,
                                    experiment_dir=experiment_dir,
                                    epsilon_start=1.0,
                                    epsilon_end=0.1,
                                    epsilon_decay_steps=500000,
                                    discount_factor=0.99,
                                    batch_size=32):

        print("\nTotal Steps: {}, Last Episode Reward: {}".format(t, stats.episode_rewards[-1]))
