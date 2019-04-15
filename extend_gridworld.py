import logging
import argparse
import ipdb
from datetime import datetime
from itertools import product
from tqdm import tqdm
import numpy as np
import pickle
from IPython import embed
from ipdb import slaunch_ipdb_on_exception

from synpo.agent import *
from synpo.component import *
from synpo.utils import *
import synpo.gridworld as gridworld

from synpo.utils import mkdir, set_seed


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', default=0, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--weight', default=None, type=str)
parser.add_argument('--scene', default=None, type=int) # use for evaluate
parser.add_argument('--task',  default=None, type=int)
parser.add_argument('--embedding_dim', default=128, type=int)
parser.add_argument('--scene_embedding_dim', default=128, type=int)
parser.add_argument('--task_embedding_dim', default=128, type=int)
parser.add_argument('--num_obj_types', default=5, type=int)
parser.add_argument('--task_length',   default=2, type=int)
parser.add_argument('--update_interval', default=1, type=int)
parser.add_argument('--scene_num', default=20, type=int)
parser.add_argument('--task_num', default=20, type=int)
parser.add_argument('--reward_prediction', action='store_false')
parser.add_argument('--split_filepath', default=None, type=str)
parser.add_argument('--wd', action='store_true')
parser.add_argument('--scene_prediction', default=1, type=int)
parser.add_argument('--task_prediction', default=1, type=int)
parser.add_argument('--option', default='normal', choices=['less', 'normal', 'more'])
parser.add_argument('--normalize_embedding', action='store_true')

parser.add_argument('--network', default='synpo', choices=['mlp', 'mtl', 'synpo'])
parser.add_argument('--postfix', default=0, type=str)
parser.add_argument('--repeat', default=10, type=int)
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--random_seed', default=0, type=int)
parser.add_argument('--logger_name', default='log/synpo_{}_{}_{}_{}.log', type=str)
parser.add_argument('--norm', action='store_true')
parser.add_argument('--y_norm', action='store_true')
parser.add_argument('--setting', type=int, default=2)
parser.add_argument('--one_traj', action='store_true')
args = parser.parse_args()

def get_network(task):
  arg_dim = task.env.observation_space.spaces[1].shape[0]
  grid_dim = task.env.observation_space.spaces[0].shape[0]
  action_dim = task.env.action_space.n
  if args.network == 'mlp':
    network = GridWorldMLP(grid_dim, action_dim, arg_dim, 
                    scene_num=args.scene_num,
                    task_num=args.task_num,
                    embed_dim=args.embedding_dim, 
                    scene_dim=args.scene_embedding_dim, 
                    task_dim=args.task_embedding_dim,
                    gpu=args.gpu_id, 
                    scene_disentanglement=args.scene_disentanglement, 
                    task_disentanglement=args.task_disentanglement,
                    norm=args.norm)
  elif args.network == 'mtl':
    network = GridWorldMTL(grid_dim, action_dim, arg_dim, 
                    scene_num=args.scene_num,
                    task_num=args.task_num,
                    embed_dim=args.embedding_dim,
                    scene_dim=args.scene_embedding_dim, 
                    task_dim=args.task_embedding_dim,
                    gpu=args.gpu_id, 
                    scene_disentanglement=args.scene_disentanglement, 
                    task_disentanglement=args.task_disentanglement,
                    norm=args.norm)
  elif args.network == 'synpo':
    network = GridWorldSynPo(grid_dim, action_dim, arg_dim, 
                    scene_num=args.scene_num,
                    task_num=args.task_num,
                    embed_dim=args.embedding_dim,
                    scene_dim=args.scene_embedding_dim,
                    task_dim=args.task_embedding_dim,
                    gpu=args.gpu_id,
                    norm=args.norm)
  else:
    raise ValueError('Non-supported Network')
  return network

def gridworld_behaviour_cloning(args, layouts, train_combos, test_combos):
  config = Config()
  grid_world_task = GridWorldTask(layouts,
                                  num_obj_types=args.num_obj_types,
                                  task_length=args.task_length,
                                  history_length= config.history_length,
                                  train_combos=train_combos,
                                  test_combos=test_combos)
  config.task_fn = lambda: grid_world_task
  if args.wd: 
    print('with weight decay!')
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.001, weight_decay=10e-5)
  else:
    print('without weight decay!')
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.001)
  network = get_network(grid_world_task)
  if args.weight is not None:
    weight_to_resume = torch.load(args.weight, map_location=lambda storage, loc: storage)['best_model_weight']
    config.extend = copy.deepcopy(weight_to_resume)
    if args.normalize_embedding:
      for k, v in weight_to_resume.items():
        if 'embed' in k:
          weight_to_resume[k] = F.normalize(v)
    network.load_state_dict(weight_to_resume)
    for k, v in network.named_parameters():
      if args.option == 'more': 
        if 'embed' not in k and 'refc' not in k and 'reward_fc' not in k and 'policy_fc' not in k: v.requires_grad = False
      elif args.option == 'normal':
        if 'embed' not in k and 'refc' not in k: v.requires_grad = False
      elif args.option == 'less': 
        if 'embed' not in k: v.requires_grad = False 
        else: print('with grad: {}'.format(k))
      else:
        raise Exception('unsupported option')
  print(network)

  config.network_fn = lambda: network
  config.replay_fn  = lambda: TrajectoryReplay(memory_size=10000, max_length=200, batch_size=64)
  config.policy_fn  = lambda: GreedyPolicy(epsilon=0.1, final_step=500000, min_epsilon=0.0)
  config.logger = Logger('./log', logger)
  config.test_interval = 2000
  config.max_eps = 10000 # 6000
  config.exploration_steps = 50000
  config.postfix = args.postfix
  config.tag = network.__class__.__name__
  config.update_interval = 1 # preset
  config.one_traj = args.one_traj
  return GridBehaviourCloning(config)

if __name__ == '__main__':
  mkdir('data')
  mkdir('data/video')
  mkdir('log')
  os.system('export OMP_NUM_THREADS=1')

  set_seed(args.random_seed, c=args.random_seed)
  layouts = ['map{}'.format(i) for i in range(1, 21) ]

  if args.setting == 2:
    train_combos = [(i, j) for i, j in product(range(10, args.scene_num), range(10, args.task_num))]
  elif args.setting == 3:
    train_combos = [(i, j) for i, j in product(range(10), range(10, args.task_num))] + [(i, j) for i, j in product(range(10, args.scene_num), range(10))]
  else:
    raise Exception('error')
  # train_combos  = [(i, j) for i, j in product(range(args.scene_num), range(args.task_num))]
  test_combos  = [(i, j) for i, j in product(range(10, args.scene_num), range(10, args.task_num))]

  agent = gridworld_behaviour_cloning(args, layouts, train_combos, test_combos)

  agent.reward_prediction = args.reward_prediction
  if args.split_filepath is None: # Default Multi-task Setting
    agent.split_name = 'MTL'
  else:
    agent.split_name = "-".join(args.split_filepath.split('/')[-2:])
  if args.evaluate:
    if args.scene is not None and args.task is not None:
      for _ in tqdm(range(args.repeat)):
        success, traj_len = agent.evaluate(visualize=args.visualize, index=(args.scene, args.task)) # main program
    else:
      rates = []
      # for combo in test_combos:
      for combo in test_combos:
        success_list = []
        trajectory_list = []
        for _ in tqdm(range(args.repeat)):
          success, traj_len, _ = agent.evaluate(visualize=args.visualize, index=combo) # main program
          success_list.append(success)
          trajectory_list.append(traj_len)
        success_rate = sum(success_list) / len(success_list)
        rates.append(success_rate)
        print('* [Task={}, # of Tests={}] Average success rate: {:.4f}, Average trajectory length: {}'.format( combo, args.repeat,
                                success_rate, sum(trajectory_list) / len(trajectory_list) ))
      print('average success rate: {:.4f}'.format(np.mean(rates)))
  else:
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(args.logger_name.format(agent.__class__.__name__,
                                                      agent.learning_network.__class__.__name__,
                                                      datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                                                      args.postfix))
    logger.addHandler(handler)
    with slaunch_ipdb_on_exception():
      train_agent(agent) # main program

