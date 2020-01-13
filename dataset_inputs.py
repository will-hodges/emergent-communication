import contextlib
import random
from collections import defaultdict
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import models
import vision
import util
from data import ShapeWorld
import data
from shapeworld import SHAPES, COLORS
from run import run
from train import init_metrics

datasets = ['single','chairs','colors']
files = ['./data/single/random/reference-1000.npz','./data/chairs/data_1000_83.npz','./data/colors/data_1000_45.npz']
for file, dataset in zip(files, datasets):
    vocab = torch.load('./models/'+dataset+'/vocab.pt')
    print(vocab)
    print(file)
    d = data.load_raw_data(file)
    dataloader = DataLoader(ShapeWorld(d, vocab), batch_size=1, shuffle=False)
    for game, (img, y, lang) in enumerate(dataloader):
        if game > (989):
            seq = []
            for word_index in lang[0,:].cpu().numpy():
                try:
                    seq.append(vocab['i2w'][word_index])
                except:
                    seq.append('<UNK>')
            plt.imsave('./output/'+dataset+'/game-'+str(game-990)+'-img-1.png', img[0][0].permute(1,2,0).cpu().numpy())
            plt.imsave('./output/'+dataset+'/game-'+str(game-990)+'-img-2.png', img[0][1].permute(1,2,0).cpu().numpy())
            plt.imsave('./output/'+dataset+'/game-'+str(game-990)+'-img-3.png', img[0][2].permute(1,2,0).cpu().numpy())
            np.savetxt('./output/'+dataset+'/game-'+str(game-990)+'-y.txt', y)
            np.savetxt('./output/'+dataset+'/game-'+str(game-990)+'-lang.txt', seq, delimiter=" ", fmt="%s")