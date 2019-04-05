# [**Syn**thesized **Po**licies (SynPO)](https://sites.google.com/view/neurips2018-synpo/home)

This repository implements the SynPo algorithms presented in:

- Hu, Hexiang, Liyu Chen, Boqing Gong, and Fei Sha. "Synthesize Policies for Transfer and Adaptation across Tasks and Environments." In Advances in Neural Information Processing Systems, pp. 1176-1185. 2018.

## Requirements

- Python 3+
- Numpy 1.10.0+
- Pytorch 0.4.0

## Usage

We release our source code based on the gridworld environment in this [repo](https://github.com/Sha-Lab/gridworld). The usage is listed as following:

```bash
usage: train_gridworld.py [-h] [--gpu_id GPU_ID] [--batch_size BATCH_SIZE]
                          [--weight WEIGHT] [--scene SCENE] [--task TASK]
                          [--embedding_dim EMBEDDING_DIM]
                          [--task_embedding_dim TASK_EMBEDDING_DIM]
                          [--num_obj_types NUM_OBJ_TYPES]
                          [--task_length TASK_LENGTH]
                          [--update_interval UPDATE_INTERVAL]
                          [--scene_num SCENE_NUM] [--task_num TASK_NUM]
                          [--reward_prediction REWARD_PREDICTION]
                          [--scene_disentanglement SCENE_DISENTANGLEMENT]
                          [--task_disentanglement TASK_DISENTANGLEMENT]
                          --split_filepath SPLIT_FILEPATH [--l L] [--lr LR]
                          [--wd] [--vc] [--mode {cloning}]
                          [--network {mlp,mtl,synpo}] [--postfix POSTFIX]
                          [--repeat REPEAT] [--evaluate] [--visualize]
                          [--random_seed RANDOM_SEED]
                          [--logger_name LOGGER_NAME] [--norm] [--y_norm]
                          [--hm_name HM_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --gpu_id GPU_ID
  --batch_size BATCH_SIZE
  --weight WEIGHT
  --scene SCENE
  --task TASK
  --embedding_dim EMBEDDING_DIM
  --task_embedding_dim TASK_EMBEDDING_DIM
  --num_obj_types NUM_OBJ_TYPES
  --task_length TASK_LENGTH
  --update_interval UPDATE_INTERVAL
  --scene_num SCENE_NUM
  --task_num TASK_NUM
  --reward_prediction REWARD_PREDICTION
  --scene_disentanglement SCENE_DISENTANGLEMENT
  --task_disentanglement TASK_DISENTANGLEMENT
  --split_filepath SPLIT_FILEPATH
  --lr LR
  --wd
  --vc
  --mode {cloning}
  --network {mlp,mtl,synpo}
  --postfix POSTFIX
  --repeat REPEAT
  --evaluate
  --visualize
  --random_seed RANDOM_SEED
  --logger_name LOGGER_NAME
  --norm
  --y_norm
  --hm_name HM_NAME
```

## References

If you are using any resources within this repo for your research, please cite:

```
@inproceedings{hu2018synthesize,
  title={Synthesized Policies for Transfer and Adaptation across Tasks and Environments},
  author={Hu, Hexiang and Chen, Liyu and Gong, Boqing and Sha, Fei},
  booktitle={Advances in Neural Information Processing Systems},
  pages={1176--1185},
  year={2018}
}
```

## License
SynPo is MIT licensed, as found in the LICENSE file.


