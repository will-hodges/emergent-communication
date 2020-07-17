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

vocab = torch.load('./models/single/vocab.pt')
print(vocab)

# Optimization
optimizer = None
loss = nn.CrossEntropyLoss()

# Metrics
metrics = init_metrics()
   
batch_size = 100
files = ['./data/single/random/reference-1000.npz','./data/single/both-needed/reference-1000.npz', './data/single/either-ok/reference-1000.npz','./data/single/shape-needed/reference-1000.npz','./data/single/color-needed/reference-1000.npz']
output_files = ['./output/single/random/new_','./output/single/both-needed/new_','./output/single/either-ok/new_','./output/single/shape-needed/new_','./output/single/color-needed/new_']
epoch = 0
listener_names = ['train','val','test']
listeners = ['./models/single/pretrained_listener_0.pt','./models/single/pretrained_listener_1.pt','./models/single/pretrained_listener_2.pt']
models = ['s','s_context','rsa','srr5','REINFORCE']
speakers = ['./models/single/literal_speaker.pt','./models/single/literal_speaker_contextual.pt',
            './models/single/literal_speaker.pt','./models/single/literal_speaker.pt',
            './models/single/pretrained_speaker_multinomial.pt']


run_types = ['sample','sample','rsa','sample','pragmatic']
activations = ['gumbel','gumbel','gumbel','gumbel','multinomial']
num_samples = [1,1,1,5,1]
for listener, listener_name in zip(listeners[2:], listener_names[2:]):
    listener = torch.load(listener)
    for i, model, speaker, run_type, activation, n in zip(list(range(7)),models[1:],speakers[1:],run_types[1:],activations[1:],num_samples[1:]):
        print(listener_name+' '+model)
        speaker = torch.load(speaker)
        for (file, output_file) in zip(files,output_files):
            print(model, run_type, activation, n)
            if model == 'no_listener':
                no_listener = True
            else:
                no_listener = False
            metrics, outputs = run(epoch, [file], 'test', run_type, speaker, listener, optimizer, loss, vocab, batch_size, True, num_samples = n, activation = activation, no_listener = no_listener)
            preds = outputs['pred'][-1].cpu().numpy()
            scores = outputs['score'][-1].cpu().numpy()
            langs = outputs['lang'][-1].cpu().numpy()
            all_seq = []
            #for game in range(langs.shape[0]-10,langs.shape[0]):
            for game in range(langs.shape[0]):
                seq = []
                for word_index in langs[game]:
                    seq.append(vocab['i2w'][word_index])
                all_seq.append(seq)
            """
            for game in range(90,langs.shape[0]):
                np.savetxt(output_file+'game_'+str(game)+'_'+str(model)+'_'+str(listener_name)+'_lang.txt', seq, delimiter=" ", fmt="%s")"""
            np.savetxt(output_file+str(model)+'_'+str(listener_name)+'_all_lang.txt', all_seq, delimiter=" ", fmt="%s")
            np.savetxt(output_file+str(model)+'_'+str(listener_name)+'_pred.txt', preds[90:])
            np.savetxt(output_file+str(model)+'_'+str(listener_name)+'_score.txt', scores[90:])
            np.save(output_file+str(model)+'_'+str(listener_name)+'_metrics.npy', metrics)
            print(metrics)