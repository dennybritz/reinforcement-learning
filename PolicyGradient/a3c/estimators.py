import numpy as np
import tensorflow as tf

class PolicyEstimator():
    """
    Policy Function approximator.
    """

    def __init__(self, num_outputs, reuse=False):
        self.num_outputs = num_outputs

        # Placeholders for our input
        # Our input are 4 RGB frames of shape 160, 160 each
        self.states = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        # The TD target value
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        X = tf.to_float(self.states) / 255.0
        batch_size = tf.shape(self.states)[0]

        # Graph shared with Value Net
        with tf.variable_scope("shared", reuse=reuse):
            # Three convolutional layers
            conv1 = tf.contrib.layers.conv2d(
                X, 32, 8, 4, activation_fn=tf.nn.relu)
            conv2 = tf.contrib.layers.conv2d(
                conv1, 64, 4, 2, activation_fn=tf.nn.relu)
            conv3 = tf.contrib.layers.conv2d(
                conv2, 64, 3, 1, activation_fn=tf.nn.relu)

        with tf.variable_scope("policy_net"):
            flattened = tf.contrib.layers.flatten(conv3)
            fc1 = tf.contrib.layers.fully_connected(flattened, 512)
            self.logits = tf.contrib.layers.fully_connected(fc1, num_outputs)
            self.probs = tf.nn.softmax(self.logits)

            self.predictions = {
                "logits": self.logits,
                "probs": self.probs
            }

            # Get the predictions for the chosen actions only
            gather_indices = tf.range(batch_size) * tf.shape(self.probs)[1] + self.actions
            self.picked_action_probs = tf.gather(tf.reshape(self.probs, [-1]), gather_indices)

            self.losses = -tf.log(self.picked_action_probs) * self.targets
            self.loss = tf.reduce_mean(self.losses)

            self.summaries = tf.merge_summary([
                tf.scalar_summary("policy_net_loss", self.loss)
            ])

            # Optimizer Parameters from original paper
            self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
            self.train_op = tf.contrib.layers.optimize_loss(
                loss=self.loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=0.00025,
                optimizer=self.optimizer,
                clip_gradients=5.0,
                summaries=tf.contrib.layers.optimizers.OPTIMIZER_SUMMARIES)



class ValueEstimator():
    """
    Value Function approximator.
    """

    def __init__(self, reuse=False):
        # Placeholders for our input
        # Our input are 4 RGB frames of shape 160, 160 each
        self.states = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        # The TD target value
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="y")

        X = tf.to_float(self.states) / 255.0
        batch_size = tf.shape(self.states)[0]

        # Graph shared with Value Net
        with tf.variable_scope("shared", reuse=reuse):
            # Three convolutional layers
            conv1 = tf.contrib.layers.conv2d(
                X, 32, 8, 4, activation_fn=tf.nn.relu)
            conv2 = tf.contrib.layers.conv2d(
                conv1, 64, 4, 2, activation_fn=tf.nn.relu)
            conv3 = tf.contrib.layers.conv2d(
                conv2, 64, 3, 1, activation_fn=tf.nn.relu)

        with tf.variable_scope("value_net"):
            flattened = tf.contrib.layers.flatten(conv3)
            fc1 = tf.contrib.layers.fully_connected(flattened, 512)
            self.logits = tf.contrib.layers.fully_connected(fc1, 1)
            self.logits = tf.squeeze(self.logits, squeeze_dims=[1])

            self.losses = tf.squared_difference(self.logits, self.targets)
            self.loss = tf.reduce_mean(self.losses)

            self.predictions = {
                "logits": self.logits
            }

            # Optimizer Parameters from original paper
            self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
            self.train_op = tf.contrib.layers.optimize_loss(
                loss=self.loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=0.00025,
                optimizer=self.optimizer,
                clip_gradients=5.0,
                summaries=tf.contrib.layers.optimizers.OPTIMIZER_SUMMARIES)

            # Summaries
            max_value = tf.reduce_max(self.logits)
            self.summaries = tf.merge_summary([
                tf.scalar_summary("value_net_loss", self.loss),
                tf.scalar_summary("max_value", max_value)
            ])