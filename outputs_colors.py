import contextlib
import random
from collections import defaultdict
import copy
import time

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

get_inputs = False

vocab = torch.load('./models/colors/vocab.pt')
print(vocab)

# Optimization
optimizer = None
loss = nn.CrossEntropyLoss()

# Metrics
metrics = init_metrics()
   
batch_size = 100
files = ['./data/colors/data_all.npz','./data/colors/data_close.npz','./data/colors/data_far.npz']
output_files = ['./output/colors/all/','./output/colors/close/','./output/colors/far/']
epoch = 0
listener_names = ['train','val','test']
listeners = ['./models/colors/pretrained-listener-0.pt','./models/colors/pretrained-listener-1.pt','./models/colors/pretrained-listener-2.pt']
models = ['s_context','s','srr5','human','pretrained_len']
speakers = ['./models/colors/literal-speaker-contextual.pt','./models/colors/literal-speaker.pt',
            './models/colors/literal-speaker.pt','./models/colors/literal-speaker.pt','./models/colors/pretrained_len_01_speaker.pt']
run_types = ['sample','sample','sample','oracle','pragmatic']
activations = ['gumbel','gumbel','gumbel','gumbel','gumbel']
num_samples = [1,1,5,1,1]
for listener, listener_name in zip(listeners[2:], listener_names[2:]):
    count = 0
    listener = torch.load(listener)
    for i, model, speaker, run_type, n in zip(list(range(6)),models,speakers,run_types,num_samples):
        print(listener_name+' '+model)
        speaker = torch.load(speaker)
        for (file, output_file) in zip(files,output_files):
            metrics, outputs = run(epoch, [file], 'val', run_type, speaker, listener, optimizer, loss, vocab, batch_size, True, num_samples = n, get_outputs = True, activation = 'gumbel', ci = False, dataset = 'colors')
            preds = outputs['pred'][-1].cpu().numpy()
            scores = outputs['score'][-1].cpu().numpy()
            langs = outputs['lang'][-1].cpu().numpy()
            for game in range(langs.shape[0]-10,langs.shape[0]):
                seq = []
                for word_index in langs[game]:
                    seq.append(vocab['i2w'][word_index])
                np.savetxt(output_file+'game_'+str(game)+'_'+str(model)+'_'+str(listener_name)+'_lang.txt', seq, delimiter=" ", fmt="%s")
            np.savetxt(output_file+str(model)+'_'+str(listener_name)+'_pred.txt', preds[90:])
            #print(output_file+str(model)+'_'+str(listener_name)+'_pred.txt', preds[90:])
            np.savetxt(output_file+str(model)+'_'+str(listener_name)+'_score.txt', scores[90:])
            #print(metrics)
            np.save(output_file+str(model)+'_'+str(listener_name)+'_metrics.npy', metrics)
            print(np.array(metrics['prob']).mean())