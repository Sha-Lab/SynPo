from itertools import product
import argparse
import pickle
import numpy as np
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=9487, type=int)
parser.add_argument('--scene_num', default=20, type=int)
parser.add_argument('--task_num', default=20, type=int)
parser.add_argument('--cover', default=4, type=int)
parser.add_argument('--num_train', default=144, type=int)
parser.add_argument('--split_filepath', default='data/hard.split', type=str)

args = parser.parse_args()
np.random.seed(args.seed)

def main(args):
  def sanity_check(train_combos, scene_num, task_num, num=4):
    from collections import defaultdict
    row = defaultdict(list)
    col = defaultdict(list)
    for i in range(scene_num):
      row[i] = []
    for i in range(task_num):
      col[i] = []
    for combo in train_combos:
      row[combo[0]].append(combo)
      col[combo[1]].append(combo)
    for k, v in row.items():
      if len(v) < num: return False
    for k, v in col.items():
      if len(v) < num: return False

    return True

  layouts = ['map{}'.format(i) for i in range(0, 20) ]
  total_combos = [(i, j) for i, j in product(range(args.scene_num), range(args.task_num))]
  if args.num_train == -1:
    args.num_train = args.scene_num*args.cover+args.task_num*args.cover-args.cover*args.cover
  while True:
    train_split = np.random.choice(range(args.scene_num*args.task_num),
                                   args.num_train,
                                   replace=False).tolist()
    train_combos = [ total_combos[idx] for idx in train_split ]
    if sanity_check(train_combos, args.scene_num, args.task_num, num=args.cover): break

  test_combos  = list(set(total_combos) - set(train_combos))

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
