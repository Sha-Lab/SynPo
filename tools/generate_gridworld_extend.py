from itertools import product
import argparse
import pickle
import numpy as np
from pprint import pprint


parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=9487, type=int)
parser.add_argument('--scene_num', default=20, type=int)
parser.add_argument('--task_num', default=20, type=int)
parser.add_argument('--extend_num', default=10, type=int)
parser.add_argument('--split_filepath', default='data/hard_extend.split', type=str)

args = parser.parse_args()
np.random.seed(args.seed)

def main(args):
  layouts = ['map{}'.format(i) for i in range(0, 20) ]
  train_combos = list(product(range(args.extend_num), range(args.extend_num)))
  test_combos = list(set(product(range(args.scene_num), range(args.task_num))) - set(train_combos))

  print('Training combos')
  pprint(train_combos)
  print('Testing combos')
  pprint(test_combos)

  table = np.zeros((args.scene_num, args.task_num))
  for i, j in train_combos:
    table[i][j] = 1

  print('Dumping splits to {}'.format(args.split_filepath))
  with open(args.split_filepath, 'wb') as handle:
    pickle.dump({'train_combos': train_combos, 'test_combos': test_combos, 'scene_num': args.scene_num, 'task_num': args.task_num, 'layouts': layouts}, handle)

if __name__ == '__main__':
  main(args)
