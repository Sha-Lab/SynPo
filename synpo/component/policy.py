import numpy as np

from synpo.utils import argmax1d

class GreedyPolicy:
  def __init__(self, epsilon, final_step, min_epsilon):
    self.init_epsilon = self.epsilon = epsilon
    self.current_steps = 0
    self.min_epsilon = min_epsilon
    self.final_step = final_step

  def sample(self, action_value, train=True):
    if train:
      if np.random.rand() < self.epsilon:
        return np.random.randint(0, len(action_value))
      return argmax1d(action_value, True)
    return np.argmax(action_value)

  def update_epsilon(self):
    diff = float(self.current_steps) / self.final_step * (self.init_epsilon - self.min_epsilon)
    self.epsilon = self.init_epsilon - diff
    self.epsilon = max(self.epsilon, self.min_epsilon)
    self.current_steps += 1
