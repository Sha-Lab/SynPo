# [**Syn**thesized **Po**licies (SynPo)](https://sites.google.com/view/neurips2018-synpo/home)

This repository implements the SynPo algorithms presented in:

- Hu, Hexiang, Liyu Chen, Boqing Gong, and Fei Sha. "Synthesized Policies for Transfer and Adaptation across Tasks and Environments." In Advances in Neural Information Processing Systems, pp. 1176-1185. 2018.

## Requirements

- Python 3+
- Numpy 1.10.0+
- Pytorch 0.4.0

Please see [requirements.txt](https://github.com/Sha-Lab/SynPo/blob/master/requirements.txt) for complete details

## Usage

We release our source code based on the gridworld environment in this [repo](https://github.com/Sha-Lab/gridworld). The usage is listed as following:

```bash
usage: train_gridworld.py [-h] [--gpu_id GPU_ID] [--batch_size BATCH_SIZE]
                          [--weight WEIGHT] [--scene SCENE] [--task TASK]
                          [--embedding_dim EMBEDDING_DIM]
                          [--scene_embedding_dim SCENE_EMBEDDING_DIM]
                          [--task_embedding_dim TASK_EMBEDDING_DIM]
                          [--num_obj_types NUM_OBJ_TYPES]
                          [--task_length TASK_LENGTH]
                          [--update_interval UPDATE_INTERVAL]
                          [--scene_num SCENE_NUM] [--task_num TASK_NUM]
                          [--reward_prediction REWARD_PREDICTION]
                          [--scene_disentanglement SCENE_DISENTANGLEMENT]
                          [--task_disentanglement TASK_DISENTANGLEMENT]
                          --split_filepath SPLIT_FILEPATH [--lr LR] [--wd]
                          [--mode {cloning}] [--network {mlp,mtl,synpo}]
                          [--postfix POSTFIX] [--repeat REPEAT] [--evaluate]
                          [--visualize] [--random_seed RANDOM_SEED]
                          [--logger_name LOGGER_NAME] [--norm]

optional arguments:
  -h, --help            show this help message and exit
  --gpu_id GPU_ID
  --batch_size BATCH_SIZE
  --weight WEIGHT
  --scene SCENE
  --task TASK
  --embedding_dim EMBEDDING_DIM
  --scene_embedding_dim SCENE_EMBEDDING_DIM
  --task_embedding_dim TASK_EMBEDDING_DIM
  --num_obj_types NUM_OBJ_TYPES
  --task_length TASK_LENGTH
  --update_interval UPDATE_INTERVAL
  --scene_num SCENE_NUM
  --task_num TASK_NUM
  --reward_prediction REWARD_PREDICTION
                        loss weight of reward prediction objective
  --scene_disentanglement SCENE_DISENTANGLEMENT
                        loss weight of scene disentanglement prediction
                        objective
  --task_disentanglement TASK_DISENTANGLEMENT
                        loss weight of task disentanglement prediction
                        objective
  --split_filepath SPLIT_FILEPATH
                        train/test split filepath
  --lr LR               base learning rate
  --wd                  enable weight decay
  --mode {cloning}      training mode [only behavior cloing available for now]
  --network {mlp,mtl,synpo}
                        select model architecture
  --postfix POSTFIX     postfix to the log file
  --repeat REPEAT       number of test run
  --evaluate            evaluation mode
  --visualize           visualize policy [only in evaluation mode]
  --random_seed RANDOM_SEED
                        random seed value
  --logger_name LOGGER_NAME
                        logger name format [must have for slots to fill]
  --norm                whether normalize the scene/task embedding
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

## Acknolwedgement
Part of the source code is modified based on the pytorch [DeepRL](https://github.com/ShangtongZhang/DeepRL) repo. We thank the original author for open source their implementation.

## License
SynPo is MIT licensed, as found in the LICENSE file.


