import numpy as np

def atari_make_initial_state(state):
    return np.stack([state] * 4, axis=2)

def atari_make_next_state(state, next_state):
    return np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)