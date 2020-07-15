import os
import sys
import copy
import json
import torch
import pickle
import random
import logging
import numpy as np
from tqdm import tqdm
from itertools import chain
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from chairs import ChairsInContext

import dataset as shapeglot

dataset = 'shapeglot'
data_dir = './'
data_size = None
image_size = 64
override_vocab = None
context_condition = 'all'
split_mode = 'easy'

w2i = shapeglot.load_pkl_file('shapeglot/language/vocab_all.pkl')
i2w = shapeglot.invert_dict(w2i)

vocab = {'w2i' : w2i, 'i2w' : i2w}

torch.save(vocab, './models/shapeglot/vocab.pt')

if dataset == 'chairs':
    DatasetClass = ChairsInContext
elif dataset == 'colors':
    DatasetClass = ColorsInContext
else:
    DatasetClass = ChairsInContext
data = DatasetClass('./'+str(dataset), image_size = 64, vocab = vocab, split = 'val', context_condition = 'far', train_frac = .95, val_frac = .05, image_transform = None)

all_imgs = None
all_labels = None
langs = None
count = 0

for i, (img, y, lang) in enumerate(data):
    label = [0, 0, 0]
    label[y] = 1
    if all_imgs is None:
        all_imgs = np.array([img.numpy()])
        all_labels = np.array([label])
        langs = np.array([lang.numpy()])
    else:
        all_imgs = np.append(all_imgs,np.array([img.numpy()]),0)
        all_labels = np.append(all_labels,np.array([label]),0)
        langs = np.append(langs,np.array([lang.numpy()]),0)
    seq = []
    if (i+1)%1000 == 0:
        print("Saving...")
        print(langs)
        data_dict = {'imgs': all_imgs, 'labels': all_labels, 'langs': langs}
        np.savez_compressed('./data/'+str(dataset)+'/data_1000_'+str(count)+'.npz', **data_dict)
        count += 1
        all_imgs = None
        all_labels = None
        langs = None