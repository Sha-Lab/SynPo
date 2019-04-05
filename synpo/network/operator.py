import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from IPython import embed

# Warpped Layer for global average pooling
class GlobalAveragePool(nn.Module):
  def forward(self, x):
    N, C = x.size(0), x.size(1)
    return x.view(N, C, -1).mean(2)

def init_weights(module):
  for layer in module.children():
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
      nn.init.xavier_uniform(layer.weight.data)
      nn.init.constant(layer.bias.data, 0)

def conv3x3(in_planes, out_planes, stride=1, groups=1):
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, groups=groups, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
  expansion = 1
  def __init__(self, in_planes, planes, stride=1):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(in_planes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion * planes:
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                      stride=stride, bias=False),
            nn.BatchNorm2d(self.expansion * planes)
        )

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += self.shortcut(x)
    out = F.relu(out)
    return out

class GridResNet(nn.Module):
  def __init__(self, block, num_blocks, in_channels, nf):
    super(GridResNet, self).__init__()
    self.in_planes = nf
    self.in_channels = in_channels

    self.datanorm = nn.BatchNorm2d(in_channels, affine=False)
    self.conv1 = conv3x3(in_channels, nf * 1, groups=4)
    self.bn1 = nn.BatchNorm2d(nf * 1)
    self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
    self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

  def _make_layer(self, block, planes, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, stride))
      self.in_planes = planes * block.expansion
    return nn.Sequential(*layers)

  def forward(self, x):
    bsz = x.size(0)
    out = F.relu(self.bn1(self.conv1(self.datanorm(x))))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    return out.view(out.size(0), -1)

class PolicyBasis(nn.Module):
  def __init__(self, action_num, state_dim, task_dim):
    super(PolicyBasis, self).__init__()
    self.state_dim    = state_dim
    self.task_dim     = task_dim
    self.action_num   = action_num

    self.weight_mu = nn.Parameter(torch.Tensor(action_num, state_dim, task_dim))
    self.policy_bias_mu = nn.Parameter(torch.Tensor(action_num))
    self.reward_bias_mu = nn.Parameter(torch.Tensor(action_num))

    self.reset_parameters()

  def forward(self, input1, input2, input3):
    N = input1.size(0)
    state_action_feat = torch.mm(input1, self.weight_mu.transpose(1, 0).contiguous().view(
                                              self.state_dim, self.action_num*self.task_dim)).view(N, self.action_num, self.task_dim)

    output1 = torch.bmm(state_action_feat, input2.unsqueeze(2)).squeeze(2)
    output2 = torch.bmm(state_action_feat, input3.unsqueeze(2)).squeeze(2)

    return output1 + self.policy_bias_mu, output2 + self.reward_bias_mu, state_action_feat

  def reset_parameters(self):
    mu_range = 1 / np.sqrt(self.state_dim*self.task_dim*self.action_num)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.policy_bias_mu.data.fill_(0)
    self.reward_bias_mu.data.fill_(0)

  def __repr__(self):
    return self.__class__.__name__ + \
            '(state_featurs={}, task_features={}, action_num={})'.format(
                    self.state_dim, self.task_dim, self.action_num)

def GridWorldResNet18(in_channels, nf=128):
  return GridResNet(BasicBlock, [2, 2, 2, 2], in_channels, nf // 8)
