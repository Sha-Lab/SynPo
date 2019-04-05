from .base_network import *
from .operator import *

import torch
import itertools
import torch.nn.functional as F
import numpy as np
from IPython import embed


# Policy network for ST2
class GridWorldSynPo(nn.Module, ValueNet):
  def __init__(self, in_channels, n_actions, arg_dim, embed_dim, scene_num, task_num, \
                     gpu=-1, feat_dim=1024, task_dim=16, scene_dim=16, scene_disentanglement=0.1, task_disentanglement=0.1, norm=True):
    super(GridWorldSynPo, self).__init__()
    self.action_space   = n_actions
    self.task_dim  = task_dim
    self.scene_dim = scene_dim
    self.feat_dim  = feat_dim
    self.embed_dim = embed_dim
    self.scene_num = scene_num
    self.task_num  = task_num

    self.scene_disentangle = scene_disentanglement > 0.0
    self.task_disentangle  = task_disentanglement > 0.0
    self.norm = norm

    out_channels = feat_dim // (2*2)
    self.state_func = GridWorldResNet18(in_channels, out_channels)
    self.state_fc   = nn.Linear(feat_dim, embed_dim)
    self.policy_basis = PolicyBasis(n_actions, embed_dim, embed_dim)

    self.task_embed  = nn.Embedding(task_num, self.task_dim)
    self.scene_embed = nn.Embedding(scene_num, self.scene_dim)

    self.policy_fc1 = nn.Linear(self.task_dim + self.scene_dim, self.embed_dim*4)
    self.policy_fc2 = nn.Linear(self.embed_dim*4, self.embed_dim)

    self.reward_fc1 = nn.Linear(self.task_dim + self.scene_dim, self.embed_dim*4)
    self.reward_fc2 = nn.Linear(self.embed_dim*4, self.embed_dim)

    self.scene_refc1 = nn.Linear(embed_dim, embed_dim*4)
    self.scene_refc2 = nn.Linear(embed_dim*4, scene_dim)

    self.task_refc1 = nn.Linear(embed_dim, embed_dim*4)
    self.task_refc2 = nn.Linear(embed_dim*4, task_dim)

    BasicNet.__init__(self, gpu)

  def forward(self, xs, _scene_ids=None, _task_ids=None):
    xs   = self.variable(xs)

    # inference the state feature
    state_feat = self.state_func(xs)
    state_feat = self.state_fc(state_feat.view(state_feat.size(0), -1))

    N = state_feat.size(0)
    # Prepare scene/task weight
    task_ids  = self.variable(_task_ids,  torch.LongTensor)
    task_emb  = self.task_embed(task_ids) # normalize embedding!
    if self.norm: task_emb = F.normalize(task_emb)

    scene_ids = self.variable(_scene_ids, torch.LongTensor)
    scene_emb  = self.scene_embed(scene_ids)
    if self.norm: scene_emb = F.normalize(scene_emb)

    policy_emb = F.relu(self.policy_fc1(torch.cat([task_emb, scene_emb], 1)))
    policy_emb = self.policy_fc2(policy_emb)

    reward_emb = F.relu(self.reward_fc1(torch.cat([task_emb, scene_emb], 1)))
    reward_emb = self.reward_fc2(reward_emb)

    # Generate Task-specific action weight
    policy, reward, state_action_feat = self.policy_basis(state_feat, policy_emb, reward_emb)

    if self.scene_disentangle:
      reproject_scene = F.relu(self.scene_refc1(state_action_feat)) # ( batch_size, num_action, state_action_dim ) -> ( batch_size*num_action, state_action_dim )
      reproject_scene = self.scene_refc2(reproject_scene).view(N*self.action_space, self.scene_dim)
      #if self.norm: reproject_scene = F.normalize(reproject_scene)

      scene_score = torch.mm(reproject_scene, self.scene_embed(self.variable(range(self.scene_num), torch.LongTensor)).t()).view(N, self.action_space, self.scene_num)
    else:
      scene_score = None

    if self.task_disentangle:
      reproject_task = F.relu(self.task_refc1(state_action_feat))
      reproject_task = self.task_refc2(reproject_task).view(N*self.action_space, self.task_dim)
      #if self.norm: reproject_task = F.normalize(reproject_task)

      task_score  = torch.mm(reproject_task,  self.task_embed( self.variable(range(self.task_num),  torch.LongTensor)).t()).view(N, self.action_space, self.task_num)
    else:
      task_score = None

    return policy, reward, scene_score, task_score

# Task single tensor ST2
class GridWorldMTL(nn.Module, ValueNet):
  def __init__(self, in_channels, n_actions, arg_dim, embed_dim, scene_num, task_num, \
                     gpu=-1, feat_dim=1024, task_dim=16, scene_dim=16, norm=True):
    super(MTL, self).__init__()
    self.action_space   = n_actions
    self.task_dim  = task_dim
    self.scene_dim = scene_dim
    self.feat_dim  = feat_dim
    self.embed_dim = embed_dim
    self.scene_num = scene_num
    self.task_num  = task_num
    self.norm = norm

    out_channels = feat_dim // (2*2)
    self.state_func = GridWorldResNet18(in_channels, out_channels)
    self.state_fc   = nn.Linear(feat_dim, embed_dim)
    self.policy_reward = PolicyBasis(n_actions, embed_dim, embed_dim)

    self.task_embed  = nn.Embedding(task_num, self.task_dim)

    self.policy_fc1 = nn.Linear(self.task_dim, self.embed_dim*4)
    self.policy_fc2 = nn.Linear(self.embed_dim*4, self.embed_dim)

    self.reward_fc1 = nn.Linear(self.task_dim, self.embed_dim*4)
    self.reward_fc2 = nn.Linear(self.embed_dim*4, self.embed_dim)

    self.task_refc1 = nn.Linear(embed_dim, embed_dim*4)
    self.task_refc2 = nn.Linear(embed_dim*4, task_dim)

    BasicNet.__init__(self, gpu)

  def forward(self, xs, _scene_ids=None, _task_ids=None):
    xs   = self.variable(xs)

    # inference the state feature
    state_feat = self.state_func(xs)
    state_feat = self.state_fc(state_feat.view(state_feat.size(0), -1))

    N = state_feat.size(0)
    # Prepare scene/task weight
    task_ids  = self.variable(_task_ids,  torch.LongTensor)
    task_emb  = self.task_embed(task_ids)
    if self.norm: task_emb = F.normalize(task_emb)

    policy_emb = F.relu(self.policy_fc1(task_emb))
    policy_emb = self.policy_fc2(policy_emb)

    reward_emb = F.relu(self.reward_fc1(task_emb))
    reward_emb = self.reward_fc2(reward_emb)

    # Generate Task-specific action weight
    policy, reward, state_action_feat = self.policy_basis(state_feat, policy_emb, reward_emb)

    reproject_task = F.relu(self.task_refc1(state_action_feat))
    reproject_task = self.task_refc2(reproject_task).view(N*self.action_space, self.task_dim)
    #if self.norm: reproject_task = F.normalize(reproject_task)

    task_score  = torch.mm(reproject_task,  self.task_embed( self.variable(range(self.task_num),  torch.LongTensor)).t()).view(N, self.action_space, self.task_num)

    return policy, reward, None, task_score

class GridWorldMLP(nn.Module, ValueNet):
  def __init__(self, in_channels, n_actions, arg_dim, embed_dim, scene_num, task_num, \
                feat_dim=1024, scene_dim=16, task_dim=16, gpu=-1, scene_disentanglement=0.1, task_disentanglement=0.1, norm=True, y_norm=True):
    super(GridWorldMLP, self).__init__()
    out_channels = feat_dim // (2*2)
    self.scene_num = scene_num
    self.task_num  = task_num
    self.norm = norm
    self.y_norm = y_norm

    self.state_func = GridWorldResNet18(in_channels, out_channels)
    self.state_fc    = nn.Linear(feat_dim, embed_dim*2)
    self.scene_embed = nn.Embedding(scene_num,  scene_dim)
    self.task_embed  = nn.Embedding(task_num,   task_dim)
    self.policy_fc1 = nn.Linear(embed_dim*2 + scene_dim + task_dim, feat_dim)
    self.policy_fc2 = nn.Linear(feat_dim, n_actions)

    self.reward_fc1 = nn.Linear(embed_dim*2 + scene_dim + task_dim, feat_dim)
    self.reward_fc2 = nn.Linear(feat_dim, n_actions)

    self.scene_refc1 = nn.Linear(embed_dim*2, embed_dim*4)
    self.scene_refc2 = nn.Linear(embed_dim*4, embed_dim)

    self.task_refc1 = nn.Linear(embed_dim*2, embed_dim*4)
    self.task_refc2 = nn.Linear(embed_dim*4, embed_dim)
    self.scene_disentangle = scene_disentanglement > 0.0
    self.task_disentangle = task_disentanglement > 0.0

    BasicNet.__init__(self, gpu)

  def forward(self, xs, _scene_ids=None, _task_ids=None):
    xs   = self.variable(xs)

    task_ids  = self.variable(_task_ids,  torch.LongTensor)
    task_emb  = self.task_embed(task_ids)
    if self.norm: task_emb = F.normalize(task_emb)

    scene_ids = self.variable(_scene_ids, torch.LongTensor)
    scene_emb = self.scene_embed(scene_ids)
    if self.norm: scene_emb = F.normalize(scene_emb)

    y = self.state_func(xs)
    y = self.state_fc(y)
    y = y.view(y.shape[0], -1)
    if self.y_norm: y = F.normalize(y)
    in_feat = torch.cat([y, scene_emb, task_emb], 1)
    policy_embed = F.relu(self.policy_fc1(in_feat))
    reward_embed = F.relu(self.reward_fc1(in_feat))

    N = y.size(0)

    reproject_scene = F.relu(self.scene_refc1(y)) 
    # ( batch_size, num_action, state_action_dim ) -> ( batch_size*num_action, state_action_dim )
    reproject_scene = self.scene_refc2(reproject_scene)

    reproject_task = F.relu(self.task_refc1(y))
    reproject_task = self.task_refc2(reproject_task)

    if self.scene_disentangle:
        scene_score = torch.mm(reproject_scene, self.scene_embed(self.variable(range(self.scene_num), torch.LongTensor)).t()).view(N, self.scene_num)
    else:
        scene_score = None

    if self.task_disentangle:
        task_score  = torch.mm(reproject_task,  self.task_embed( self.variable(range(self.task_num),  torch.LongTensor)).t()).view(N, self.task_num)
    else:
        task_score = None

    return self.policy_fc2(policy_embed), self.reward_fc2(reward_embed), scene_score, task_score
