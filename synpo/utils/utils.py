import os
import shutil
import torch
import numpy as np
import random


def mkdir(path, rm=False):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if rm:
            shutil.rmtree(path)
            os.makedirs(path)

def set_seed(t, r=None, p=None, c=None):
    if r is None:
        r = t
    if p is None:
        p = r
    torch.manual_seed(t)
    random.seed(r)
    np.random.seed(p)
    if c is not None:
      torch.cuda.manual_seed(c)

def extract(d, *args):
    ret = []
    for k in args:
        ret.append(d[k])
    return ret

def argmax1d(a, random_tie=False):
    a = np.asarray(a)
    if random_tie:
        return np.random.choice(np.flatnonzero(a == a.max()))
    else:
        return np.argmax(a)

def discount_cumsum(xs, discount=0.99):
    r = 0.0
    res = []
    for x in xs[::-1]:
        r = r * discount + x
        res.append(r)
    return res[::-1]

