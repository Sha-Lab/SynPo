import numpy as np
import pickle
import os
import os.path as osp
import copy
from collections import defaultdict
import random

from IPython import embed
import torch
from .utils import mkdir

def success(r):
    return r > 0

def test_agent(agent, test_on_train=False):
  config = agent.config
  combo_test_rewards = defaultdict(list)
  combo_test_success = defaultdict(list)

  test_samples = agent.task.train_combos if test_on_train else agent.task.test_combos
  if config.n_test_samples:
    sampled_combos = random.choices(test_samples, k=config.n_test_samples)
  else:
    sampled_combos = test_samples
  for combo in sampled_combos:
    test_reward, _, _, _ = agent.episode(train=False, env_kwargs=dict(index=combo, sample_pos=True))
    combo_test_rewards[combo].append(test_reward)
    combo_test_success[combo].append(success(test_reward))
  config.logger.info('average test success rate and rewards of each task')
  for k, v in combo_test_rewards.items():
      config.logger.info('{} {}: {} {}'.format(k, agent.task.env.task_desc[k[-1]], np.mean(v), np.mean(combo_test_success[k])))
  avg_test_reward = np.mean(np.concatenate(list(combo_test_rewards.values())))
  avg_test_success_rate = np.mean(np.concatenate(list(combo_test_success.values())))
  return avg_test_reward, avg_test_success_rate

def train_agent(agent):
  config = agent.config
  window_size = 100
  ep = 0
  rewards = []
  steps = []
  avg_test_rewards = []
  avg_test_success_rates = []
  agent_type = agent.__class__.__name__
  best_model = None
  best_sr = 0
  combo_success = defaultdict(list)
  model_bank = []
  # plot
  name = '{}-{}-{}-{}-{}'.format(agent_type, agent.task.name, config.tag, agent.split_name, agent.config.postfix)
  while True:
    ep += 1
    reward, loss, step, index = agent.episode()
    combo_success[index].append(success(reward))
    rewards.append(reward)
    steps.append(step)
    config.logger.info('episode %d, reward %.3f, idx (%d, %d), avg loss %.3f, total steps %d, episode step %d, epislon %.2f' % (
      ep, reward, index[0], index[1], sum(loss) / (len(loss) + 1e-12), agent.total_steps, step, agent.policy.epsilon))

    if config.episode_limit and ep > config.episode_limit: break

    if config.test_interval and ep % config.test_interval == 0 and agent.total_steps > config.exploration_steps:
      config.logger.info('averge success rate of each task:')
      for k, v in combo_success.items():
          config.logger.info('{} {}: {}'.format(k, agent.task.env.task_desc[k[-1]], np.mean(v)))
      combo_success.clear()

      config.logger.info('Testing on train...')
      avg_train_reward, avg_train_success_rate = test_agent(agent, test_on_train=True)
      config.logger.info('Avg test success rate %f, Avg test reward %f' % (avg_train_success_rate, avg_train_reward))

      config.logger.info('Testing on test...')
      avg_test_reward, avg_test_success_rate = test_agent(agent)
      avg_test_rewards.append(avg_test_reward)
      avg_test_success_rates.append(avg_test_success_rate)
      if best_sr <= avg_test_success_rate:
        best_sr = avg_test_success_rate
        best_model = copy.deepcopy(agent.learning_network.state_dict())
      config.logger.info('Avg test success rate %f, Avg test reward %f' % (avg_test_success_rate, avg_test_reward))

      #==============================================
      # Unwrapped Model Saving Routine
      #==============================================
      snapshot_filepath = osp.join('data', 'outputs', '{}-{}-{}-{}-{}'.format(agent_type, agent.task.name, config.tag, agent.split_name, agent.config.postfix))
      mkdir(snapshot_filepath)
      torch.save({'best_model_weight': agent.learning_network.state_dict() }, osp.join(snapshot_filepath,
                  'episode.{}.train-sr.{:3f}.test-sr.{:3f}.train-rw.{:3f}.test-rw.{:3f}.model'.format(ep, avg_train_success_rate, avg_test_success_rate, avg_train_reward, avg_test_reward)))
      torch.save({'task': agent.task.save_config(), 'best_sr': best_sr, 'best_model_weight': best_model, 'rewards': rewards,
                  'steps': steps, 'avg_test_rewards': avg_test_rewards, 'avg_test_success_rates': avg_test_success_rates},
                  osp.join(snapshot_filepath, 'train.record'))

    if (config.max_steps and agent.total_steps > config.max_steps) or (config.max_eps and ep > config.max_eps):
      config.logger.info('Testing on train Before Fiishing...')
      avg_train_reward, avg_train_success_rate = test_agent(agent, test_on_train=True)
      config.logger.info('Avg test success rate %f, Avg test reward %f' % (avg_train_success_rate, avg_train_reward))

      config.logger.info('Testing on test Before Finishing...')
      avg_test_reward, avg_test_success_rate = test_agent(agent)
      avg_test_rewards.append(avg_test_reward)
      avg_test_success_rates.append(avg_test_success_rate)
      if best_sr <= avg_test_success_rate:
        best_sr = avg_test_success_rate
        best_model = copy.deepcopy(agent.learning_network.state_dict())
      config.logger.info('Avg test success rate %f, Avg test reward %f' % (avg_test_success_rate, avg_test_reward))
      #==============================================
      # Unwrapped Model Saving Routine
      #==============================================
      snapshot_filepath = osp.join('data', 'outputs', '{}-{}-{}-{}-{}'.format(agent_type, agent.task.name, config.tag, agent.split_name, agent.config.postfix))
      mkdir(snapshot_filepath)
      torch.save({'best_model_weight': agent.learning_network.state_dict() }, osp.join(snapshot_filepath,
                  'episode.{}.train-sr.{:3f}.test-sr.{:3f}.train-rw.{:3f}.test-rw.{:3f}.model'.format(ep, avg_train_success_rate, avg_test_success_rate, avg_train_reward, avg_test_reward)))
      torch.save({'task': agent.task.save_config(), 'best_sr': best_sr, 'best_model_weight': best_model, 'rewards': rewards,
                  'steps': steps, 'avg_test_rewards': avg_test_rewards, 'avg_test_success_rates': avg_test_success_rates},
                  osp.join(snapshot_filepath, 'train.record'))
      break
    
  os.system("ls {} > {}/ls.log".format(snapshot_filepath, snapshot_filepath)) # automatically log
  agent.close()
  return steps, rewards, avg_test_rewards, avg_test_success_rates
