import gym
import sys
import numpy as np
import os.path as osp
import multiprocessing as mp
import sys
import torch

from IPython import embed


class BasicTask:
  def __init__(self, max_steps=sys.maxsize):
    self.steps = 0
    self.max_steps = max_steps

  def reset(self, *args):
    self.steps = 0
    state = self.env.reset(*args)
    return state

  def normalize_state(self, state):
    return state

  def step(self, action):
    next_state, reward, done, info = self.env.step(action)
    self.steps += 1
    done = (done or self.steps >= self.max_steps)
    return next_state, reward, done, info

  def random_action(self):
    return self.env.action_space.sample()

class GridWorldTask(BasicTask):
  def __init__(
            self,
            layouts=['map{}'.format(i) for i in range(11, 31)],
            num_obj_types=5,
            task_length=2,
            history_length=4,
            max_steps=300,
            train_combos=None,
            test_combos=None,
            gaussian_img=True,
            l=2,
            vc=False,
            record=False,
            ):
    from synpo.gridworld.env import GridWorld, read_map, ComboEnv, PORGBEnv
    self.train_combos = train_combos
    self.test_combos  = test_combos
    self.num_combos   = len(train_combos) + len(test_combos)
    self.env = PORGBEnv(ComboEnv(GridWorld(
                              layouts,
                              window=history_length,
                              task_length=task_length,
                              num_obj_types=num_obj_types,
                              train_combos=train_combos,
                              test_combos=test_combos,
                              gaussian_img=gaussian_img)), l=l, vc=vc, record=record)
    self.action_dim = self.env.action_space.n
    self.max_steps = max_steps
    self.name = 'gridworld'

  def save_config(self):
    return self.__dict__

  def reset(self, index=None, sample_pos=True, train=True):
    self.steps = 0
    state = self.env.reset(index, sample_pos=sample_pos, train=train)
    return state[0]

  def step(self, action):
    next_state, reward, done, info = self.env.step(action)
    self.steps += 1
    done = (done or self.steps >= self.max_steps)
    return next_state[0], reward, done, info

  def normalize_state(self, state):
    return np.asarray([np.asarray(s) for s in state])

  def get_opt_action(self):
    return self.env.get_opt_action()

  def get_random_opt_action(self, discount):
    return self.env.get_random_opt_action(discount)

  def get_q(self, *args, **kwargs):
    return self.env.get_q(*args, **kwargs)

  def get_qs(self, *args, **kwargs):
    return self.env.get_qs(*args, **kwargs)

  def index(self):
    return self.env.index()

  def seed(self, *args, **kwargs):
    return self.env.seed(*args, **kwargs)

  def pos(self):
    return self.env.unwrapped.x, self.env.unwrapped.y

def sub_task(parent_pipe, pipe, task_fn):
    parent_pipe.close()
    task = task_fn()
    task.env.seed(np.random.randint(0, sys.maxsize))
    while True:
        op, data = pipe.recv()
        if op == 'step':
            pipe.send(task.step(data))
        elif op == 'reset':
            pipe.send(task.reset())
        elif op == 'exit':
            pipe.close()
            return
        else:
            assert False, 'Unknown Operation'

class ParallelizedTask:
    def __init__(self, task_fn, num_workers):
        self.task_fn = task_fn
        self.task = task_fn()
        self.name = self.task.name
        self.pipes, worker_pipes = zip(*[mp.Pipe() for _ in range(num_workers)])
        args = [(p, wp, task_fn) for p, wp in zip(self.pipes, worker_pipes)]
        self.workers = [mp.Process(target=sub_task, args=arg) for arg in args]
        for p in self.workers: p.start()
        for p in worker_pipes: p.close()
        self.observation_space = self.task.env.observation_space
        self.action_space = self.task.env.action_space

    def step(self, actions):
        for pipe, action in zip(self.pipes, actions):
            pipe.send(('step', action))
        results = [p.recv() for p in self.pipes]
        results = map(lambda x: np.stack(x), zip(*results))
        return results

    def reset(self, i=None):
        if i is None:
            for pipe in self.pipes:
                pipe.send(('reset', None))
            results = [p.recv() for p in self.pipes]
        else:
            self.pipes[i].send(('reset', None))
            results = self.pipes[i].recv()
        return np.stack(results)

    def close(self):
        for pipe in self.pipes:
            pipe.send(('exit', None))
        for p in self.workers: p.join()
