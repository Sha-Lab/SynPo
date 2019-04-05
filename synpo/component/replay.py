import numpy as np
import torch
import random
import torch.multiprocessing as mp
from collections import defaultdict
from IPython import embed

from synpo.utils import discount_cumsum

class TrajectoryReplay:
  def __init__(self, memory_size, max_length, batch_size, discount=0.99):
    self.memory_size = memory_size
    self.batch_size = batch_size
    self.max_length = max_length
    self.discount = discount

    self.clear()

  def feed(self, experience):
    state, action, reward, _, qs, _, scene_id, task_id, done = experience

    self.cur_states[self.cur_pos] = state
    self.cur_actions[self.cur_pos] = action
    self.cur_rewards[self.cur_pos] = reward
    self.cur_qs[self.cur_pos] = qs
    self.cur_scene_ids[self.cur_pos] = scene_id
    self.cur_task_ids[self.cur_pos] = task_id

    self.cur_pos += 1
    if done:
        self.feed_traj()

  def feed_traj(self):
    if self.full:
      _sid, _tid = self.scene_ids[self.pos][0], self.task_ids[self.pos][0]
      self.combo_ids[_sid, _tid].remove(self.pos)

    actual_q = np.asarray(discount_cumsum(self.cur_rewards[:self.cur_pos], self.discount))

    self.states[self.pos, :self.cur_pos] = self.cur_states[:self.cur_pos]
    self.actions[self.pos, :self.cur_pos] = self.cur_actions[:self.cur_pos]
    self.rewards[self.pos, :self.cur_pos] = self.cur_rewards[:self.cur_pos]
    self.qs[self.pos, :self.cur_pos] = self.cur_qs[:self.cur_pos]
    self.actual_q[self.pos, :self.cur_pos] = actual_q
    self.scene_ids[self.pos, :self.cur_pos] = self.cur_scene_ids[:self.cur_pos]
    self.task_ids[self.pos, :self.cur_pos] = self.cur_task_ids[:self.cur_pos]
    self.t_pos[self.pos] = self.cur_pos

    self.combo_ids[self.scene_ids[self.pos][0], self.task_ids[self.pos][0]].add(self.pos)
    self.pos += 1
    self.cur_pos = 0

    if self.pos == self.memory_size:
      self.full = True
      self.pos = 0

  def _sample(self, sampled_indices):
    return {
      'states': np.concatenate([self.states[i, :self.t_pos[i]] for i in sampled_indices]),
      'actions': np.concatenate([self.actions[i, :self.t_pos[i]] for i in sampled_indices]),
      'rewards': np.concatenate([self.rewards[i, :self.t_pos[i]] for i in sampled_indices]),
      'qs': np.concatenate([self.qs[i, :self.t_pos[i]] for i in sampled_indices]),
      'actual_q': np.concatenate([self.actual_q[i, :self.t_pos[i]] for i in sampled_indices]),
      'scene_ids': np.concatenate([self.scene_ids[i, :self.t_pos[i]] for i in sampled_indices]),
      'task_ids': np.concatenate([self.task_ids[i, :self.t_pos[i]] for i in sampled_indices]),
    }

  def sample(self):
    upper_bound = self.memory_size if self.full else self.pos
    sampled_indices = np.random.randint(0, upper_bound, size=self.batch_size)
    return self._sample(sampled_indices)

  def stratified_sample(self, combo_id=None):
    sampled_indices = []
    if combo_id is None: #Default: multi-task sampling
      sampling_cands = []
      sampling_tasks = { k[1]: [] for k in self.combo_ids.keys() }
      num_tasks = len(sampling_tasks.keys())
      for combo_id, val in self.combo_ids.items(): sampling_tasks[combo_id[1]].append(val)
      for k in sampling_tasks.keys():
        sampling_cands.extend( np.random.choice(sampling_tasks[k], round(self.batch_size / num_tasks )))
      for v in sampling_cands:
        sampled_indices.extend( random.sample(v, min(len(v), 1)) )
    else:
      sid, tid = combo_id
      if sid is not None and tid is None: # Marginalized sampling according to scene
        sampling_cands = [ v for k, v in self.combo_ids.items() if k[0] == sid ]
        sampling_cands = np.random.choice(sampling_cands, self.batch_size, replace=True)
        for v in sampling_cands:
          sampled_indices.extend( random.sample(v, min(len(v), 1)) )
      elif sid is None and tid is not None: # Marginalized sampling according to tasks
        sampling_cands = [ v for k, v in self.combo_ids.items() if k[1] == tid ]
        sampling_cands = np.random.choice(sampling_cands, self.batch_size, replace=True)
        for v in sampling_cands:
          sampled_indices.extend( random.sample(v, min(len(v), 1)) )
      else: # Specified sampling
        sampled_indices = random.sample(self.combo_ids[combo_id], self.batch_size)

    sampled_indices = np.asarray(sampled_indices)
    return self._sample(sampled_indices)

  def get_all(self):
    upper_bound = self.memory_size if self.full else self.pos
    sampled_indices = np.arange(upper_bound)
    return self._sample(sampled_indices)

  def clear(self):
    self.states = np.array([[None] * self.max_length] * self.memory_size)
    self.actions = np.empty((self.memory_size, self.max_length), dtype=np.uint8)
    self.rewards = np.empty((self.memory_size, self.max_length))
    self.qs = np.array([[None] * self.max_length] * self.memory_size)
    self.actual_q = np.empty((self.memory_size, self.max_length))
    self.scene_ids = np.empty((self.memory_size, self.max_length), dtype=np.uint8)
    self.task_ids = np.empty((self.memory_size, self.max_length), dtype=np.uint8)

    self.cur_states = np.array([None] * self.max_length)
    self.cur_actions = np.array([None] * self.max_length)
    self.cur_rewards = np.empty(self.max_length)
    self.cur_qs = np.array([None] * self.max_length)
    self.cur_scene_ids = np.empty(self.max_length, dtype=np.uint8)
    self.cur_task_ids = np.empty(self.max_length, dtype=np.uint8)
    self.cur_pos = 0

    self.combo_ids = defaultdict(set)

    self.pos = 0 # trajectory
    self.t_pos = np.zeros(self.memory_size, dtype=np.uint16) # within trajectory
    self.full = False

  def total_size(self):
    upper_bound = self.memory_size if self.full else self.pos
    return np.sum(self.t_pos[:upper_bound])
