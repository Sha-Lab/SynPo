from synpo.network import *
from synpo.component import *
from synpo.utils import *
import numpy as np
import time
import os
import pickle
import torch
from torch.nn.utils.clip_grad import clip_grad_norm
import random
import copy

from synpo.utils import argmax1d, extract

# Information profiling
def profile_action_value(action, _action, value, q_value):
  print('[{}] prediction: {}, ground truth: {} ({:.4f}/{:.4f})'.format( \
                              'FAIL' if action != _action else 'BINGO', \
                              action, _action, float(value[_action]), float(q_value)))

if torch.cuda.is_available():
  log_softmax = nn.LogSoftmax(dim=1).cuda()
else:
  log_softmax = nn.LogSoftmax(dim=1)
mse_criterion = nn.MSELoss()
cross_entropy_criterion = lambda x, y: ( -log_softmax(x) * y ).sum(dim=1).mean()

# also use qs
def imitation_loss(agent, experiences):
  states, actions, rewards, qs, scene_ids, task_ids = extract(experiences,
    'states', 'actions', 'rewards', 'qs', 'scene_ids', 'task_ids')
  states = agent.task.normalize_state(states)
  # qs = np.asarray([ q for q in  qs ])
  # act_id  = np.argmax(qs, axis=1)
  action  = agent.learning_network.variable(np.eye(5)[actions])
  agent.cached_value = agent.cached_value or agent.learning_network.predict(states, scene_ids, task_ids, False)

  loss = 0
  if isinstance(agent.cached_value, tuple) and ( isinstance(agent.learning_network, GridWorldSynPo) or \
                                                 isinstance(agent.learning_network, GridWorldMTL) or \
                                                 isinstance(agent.learning_network, GridWorldMLP) ):
    act, _, scene_scores, task_scores = agent.cached_value
    actions  = agent.learning_network.variable(actions, torch.LongTensor).view(-1, 1, 1)

    if scene_scores is not None:
      if len(scene_scores.size()) > 2: scene_scores = scene_scores.gather(1, actions.expand(-1, 1, scene_scores.size(2))).squeeze(1)
      scene_gt = agent.learning_network.variable(np.eye(scene_scores.size(1))[scene_ids])

      loss += agent.config.scene_disentanglement_coeff * cross_entropy_criterion(scene_scores, scene_gt)

    if task_scores is not None:
      if len(task_scores.size()) > 2:  task_scores = task_scores.gather(1, actions.expand(-1, 1, task_scores.size(2))).squeeze(1)
      task_gt  = agent.learning_network.variable(np.eye(task_scores.size(1))[task_ids])

      loss += agent.config.task_disentanglement_coeff * cross_entropy_criterion(task_scores,  task_gt)
  elif isinstance(agent.cached_value, tuple):
    act  = agent.cached_value[0]
  else:
    act  = agent.cached_value
  loss += cross_entropy_criterion(act, action)
  return loss

def reward_prediction_loss(agent, experiences):
  assert isinstance(agent.learning_network, ValueNet)
  states, actions, rewards, qs, scene_ids, task_ids = extract(experiences,
    'states', 'actions', 'rewards', 'qs', 'scene_ids', 'task_ids')
  states = agent.task.normalize_state(states)
  agent.cached_value = agent.cached_value or agent.learning_network.predict(states, scene_ids, task_ids, False)

  r  = agent.cached_value[1]
  rewards = agent.learning_network.variable(rewards)
  r = r[np.arange(len(actions)), actions]
  loss =  mse_criterion(r, rewards)
  return loss

class GridAgent:
  def __init__(self, config):
    self.config = config
    self.learning_network = config.network_fn()
    self.target_network = config.network_fn()
    self.target_network.load_state_dict(self.learning_network.state_dict())
    self.task = config.task_fn()
    self.replay = config.replay_fn()
    self.policy = config.policy_fn()
    self.total_steps = 0
    self.update_interval = config.update_interval
    # add reward loss or not
    self.reward_prediction = False
    # cache calculation
    self.cached_value = None

  def episode(self, train=True, env_kwargs={}):
    raise NotImplementedError('Re-Write this method.')

  def close(self):
    pass

  def evaluate(self, visualize=False, step_time=0.1, seed=None, index=None, optimal=False):
    assert index is not None, 'just because I set default to None does not mean that you can leave it None'
    if seed is not None:
      self.task.seed(seed)
    heat_map = np.zeros((16, 16)) # does not count initialized position
    rng = copy.deepcopy(self.task.env.unwrapped.random)
    actions = []
    state = self.task.reset(index, sample_pos=True)
    trajectory = []
    accum_rewards = []
    while True:
      if visualize:
        self.task.env.unwrapped.render() # change
        time.sleep(step_time)
      if optimal:
        value = self.task.get_qs(self.config.discount)
      else:
        value = self.learning_network.predict(self.task.normalize_state([state]),
                                            np.asarray([index[0]]), np.asarray([index[1]]),
                                            to_numpy=True, evaluate=True).flatten()
      action = np.argmax(value)
      actions.append(action)
      #action = self.task.get_opt_action() 
      state, reward, done, _ = self.task.step(action)
      heat_map[self.task.pos()] = 1
      trajectory.append(action)
      accum_rewards.append(reward)

      if done: break
    return (reward > 10), len(trajectory), sum(accum_rewards), heat_map, rng, actions

  def evaluate2image(self, index, seed=None):
    if seed is not None:
      self.task.seed(seed)
    state = self.task.reset(index, sample_pos=True)
    trajectory = []
    while True:
      trajectory.append(self.task.env.pretty_render().astype(np.uint8))
      value = self.learning_network.predict(self.task.normalize_state([state]),
                                            np.asarray([index[0]]), np.asarray([index[1]]),
                                            to_numpy=True, evaluate=True).flatten()
      action = np.argmax(value)
      state, reward, done, _ = self.task.step(action)

      if done: break
    return (reward > 0), len(trajectory), trajectory

class GridBehaviourCloning(GridAgent):
  def __init__(self, config):
    super(GridBehaviourCloning, self).__init__(config)
    keys = self.learning_network.state_dict().keys()
    self.optimizer = config.optimizer_fn( [ v for v in self.learning_network.parameters() if v.requires_grad ])
    self.grad_clip = config.grad_clip
    self.one_traj = config.one_traj

  def episode(self, train=True, env_kwargs={}):
    self.cached_value = None
    episode_start_time = time.time()
    state = self.task.reset(**env_kwargs)
    scene_id, task_id = self.task.env.unwrapped.index()
    total_reward = 0.0
    steps = 0
    total_loss = []
    while True:
      if not train:
        value = self.learning_network.predict(self.task.normalize_state([state]),
                                                np.asarray([scene_id]), np.asarray([task_id]),
                                                True, evaluate=True).flatten()
      qs = self.task.get_qs(discount=self.config.discount)
      if self.total_steps < self.config.exploration_steps:
        #action = self.policy.sample(qs, train=train)
        action = argmax1d(qs)
      else:
        if train:
          #action = self.policy.sample(qs, train=train)
          action = argmax1d(qs)
        else:
          action = self.policy.sample(value, train=train)
      next_state, reward, done, _ = self.task.step(action)
      total_reward += reward
      if train:
        if not self.one_traj or (scene_id, task_id) not in self.replay.combo_ids:
            self.replay.feed([state, action, reward, None, qs, next_state, scene_id, task_id, int(done)])
        self.total_steps += 1
      steps += 1
      state = next_state
      if done and train and self.total_steps > self.config.exploration_steps:
        experiences = self.replay.sample()
        if isinstance(self.learning_network, ValueNet):
          if self.reward_prediction:
            loss = imitation_loss(self, experiences) + 0.01 * reward_prediction_loss(self, experiences)
          else:
            loss = imitation_loss(self, experiences)
        else:
          raise NotImplementedError('Not supported network')
        self.optimizer.zero_grad()
        loss.backward()
        total_loss.append( loss.data.cpu().item() )
        if self.grad_clip > 0:
          clip_grad_norm(self.learning_network.parameters(), self.grad_clip)
        self.optimizer.step()
        if self.config.extend is not None:
          self.learning_network.scene_embed.weight.data[:10, :].copy_(self.config.extend['scene_embed.weight'][:10, :])
          self.learning_network.task_embed.weight.data[:10, :].copy_(self.config.extend['task_embed.weight'][:10, :])

      if train and self.total_steps > self.config.exploration_steps:
        self.policy.update_epsilon()

      if done: break

    episode_time = time.time() - episode_start_time
    self.config.logger.debug('episode steps %d, episode time %f, time per step %f' %
                      (steps, episode_time, episode_time / float(steps)))
    return total_reward, total_loss, steps, (scene_id, task_id)
