"""
A simplified version of training transformer
"""

import argparse
import math
import time
import dill as pickle
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import Field, Dataset, BucketIterator
from torchtext.datasets import TranslationDataset

import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim

# --------------------
parser = argparse.ArgumentParser()
parser.add_argument('-data_pkl', default=None)     # all-in-1 data pickle or bpe field

parser.add_argument('-train_path', default=None)   # bpe encoded data
parser.add_argument('-val_path', default=None)     # bpe encoded data

parser.add_argument('-epoch', type=int, default=10)
parser.add_argument('-b', '--batch_size', type=int, default=2048)

parser.add_argument('-d_model', type=int, default=512)
parser.add_argument('-d_inner_hid', type=int, default=2048)
parser.add_argument('-d_k', type=int, default=64)
parser.add_argument('-d_v', type=int, default=64)

parser.add_argument('-n_head', type=int, default=8)
parser.add_argument('-n_layers', type=int, default=6)
parser.add_argument('-warmup','--n_warmup_steps', type=int, default=4000)

parser.add_argument('-dropout', type=float, default=0.1)
parser.add_argument('-embs_share_weight', action='store_true')
parser.add_argument('-proj_share_weight', action='store_true')

# parser.add_argument('-log', default=None)
# parser.add_argument('-save_model', default=None)
# parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

# parser.add_argument('-no_cuda', action='store_true')
parser.add_argument('-label_smoothing', action='store_true')

opt = parser.parse_args()
opt.cuda = True if torch.cuda.is_available() else False
opt.d_word_vec = opt.d_model

# ----------------------

