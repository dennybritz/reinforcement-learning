import numpy as np

class AtariEnvWrapper(object):
  """
  Wraps an Atari environment to end an episode when a life is lost.
  """
  def __init__(self, env):
    self.env = env

  def __getattr__(self, name):
    return getattr(self.env, name)

  def step(self, *args, **kwargs):
    lives_before = self.env.ale.lives()
    next_state, reward, done, info = self.env.step(*args, **kwargs)
    lives_after = self.env.ale.lives()

    # End the episode when a life is lost
    if lives_before > lives_after:
      done = True

    # Clip rewards to [-1,1]
    reward = max(min(reward, 1), -1)

    return next_state, reward, done, info

def atari_make_initial_state(state):
  return np.stack([state] * 4, axis=2)

def atari_make_next_state(state, next_state):
  return np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)