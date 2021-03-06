import contextlib
import random
from collections import defaultdict
import copy
import os

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

vocab = torch.load('./models/single/vocab.pt')
print(vocab)

#directories = ['random','both-needed','color-needed','shape-needed','either-ok']
directories = ['generalization_new_combo','generalization_new_color','generalization_new_shape']
data_files = ['./data/single/generalization_new_combo/test/reference-1000.npz','./data/single/generalization_new_color/test/reference-1000.npz','./data/single/generalization_new_shape/test/reference-1000.npz']
for directory, data_file in zip(directories, data_files):
    #d = data.load_raw_data('./data/single/'+str(directory)+'/reference-1000.npz')
    d = data.load_raw_data(data_file)
    dataloader = DataLoader(ShapeWorld(d, vocab), batch_size=1, shuffle=False)
    for game, (img, y, lang) in enumerate(dataloader):
        if game > (989):
            plt.imsave('./output/single/'+str(directory)+'/game-'+str(game-990)+'-img-1.png', img[0][0].permute(1,2,0).cpu().numpy())
            plt.imsave('./output/single/'+str(directory)+'/game-'+str(game-990)+'-img-2.png', img[0][1].permute(1,2,0).cpu().numpy())
            plt.imsave('./output/single/'+str(directory)+'/game-'+str(game-990)+'-img-3.png', img[0][2].permute(1,2,0).cpu().numpy())
            np.savetxt('./output/single/'+str(directory)+'/game-'+str(game-990)+'-y.txt', y)
            np.savetxt('./output/single/'+str(directory)+'/game-'+str(game-990)+'-lang.txt', lang)
"""
            
vocab = torch.load('./models/colors/vocab.pt')
print(vocab)

directories = ['all','close','far']
for directory in directories:
    d = data.load_raw_data('./data/colors/data_'+str(directory)+'.npz')
    dataloader = DataLoader(ShapeWorld(d, vocab), batch_size=1, shuffle=False)
    samples = len(dataloader)
    
    for game, (img, y, lang) in enumerate(dataloader):
        if game > (samples-11):
            plt.imsave('./output/colors/'+str(directory)+'/game-'+str(game-(samples-10))+'-img-1.png', img[0][0].permute(1,2,0).cpu().numpy())
            plt.imsave('./output/colors/'+str(directory)+'/game-'+str(game-(samples-10))+'-img-2.png', img[0][1].permute(1,2,0).cpu().numpy())
            plt.imsave('./output/colors/'+str(directory)+'/game-'+str(game-(samples-10))+'-img-3.png', img[0][2].permute(1,2,0).cpu().numpy())
            np.savetxt('./output/colors/'+str(directory)+'/game-'+str(game-(samples-10))+'-y.txt', y)
            np.savetxt('./output/colors/'+str(directory)+'/game-'+str(game-(samples-10))+'-lang.txt', lang)"""