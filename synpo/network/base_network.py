import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Base class for all kinds of network
class BasicNet:
  def __init__(self, gpu, LSTM=False, stochastic=False):
    if not torch.cuda.is_available():
      gpu = -1
    self.gpu = gpu
    self.stochastic = stochastic
    self.LSTM = LSTM
    self.init_weights()
    if self.gpu >= 0:
      self.cuda(self.gpu)

  def supported_dtype(self, x, torch_type):
    if torch_type == torch.FloatTensor:
      return np.asarray(x, dtype=np.float32)
    if torch_type == torch.LongTensor:
      return np.asarray(x, dtype=np.int64)

  def variable(self, x, dtype=torch.FloatTensor, requires_grad=False):
    if isinstance(x, Variable):
      return x
    x = dtype(torch.from_numpy(self.supported_dtype(x, dtype)))
    if self.gpu >= 0:
      x = x.cuda(self.gpu)
    return Variable(x, requires_grad=requires_grad)

  def tensor(self, x, dtype=torch.FloatTensor):
    x = dtype(torch.from_numpy(self.supported_dtype(x, dtype)))
    if self.gpu >= 0:
      x = x.cuda(self.gpu)
    return x

  def reset_noise(self):
    raise NotImplementedError('Not Supported')

  def reset(self, terminal):
    if not self.LSTM:
      return
    if terminal:
      self.h.data.zero_()
      self.c.data.zero_()
    self.h = Variable(self.h.data)
    self.c = Variable(self.c.data)

  def init_weights(self):
    for layer in self.children():
      relu_gain = nn.init.calculate_gain('relu')
      if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        nn.init.xavier_uniform(layer.weight.data)
        nn.init.constant(layer.bias.data, 0)
      if isinstance(layer, nn.Embedding):
        nn.init.xavier_uniform(layer.weight.data)

# Pass both scene and task
class ValueNet(BasicNet):
  def predict(self, x, _scene_ids=None, _task_ids=None, to_numpy=False, evaluate=False):
    if evaluate:    self.eval()
    else:           self.train()
    y = self.forward(x, _scene_ids, _task_ids)
    if to_numpy:
      y = y[0]
      if type(y) is list:
        y = [y_.cpu().data.numpy() for y_ in y]
      else:
        y = y.cpu().data.numpy()
      return y
    else:
      return y
