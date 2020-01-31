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
output_files = ['./output/single/random/','./output/single/both-needed/','./output/single/either-ok/','./output/single/shape-needed/','./output/single/color-needed/']
epoch = 0
"""
print('oracle')
speaker = None
listener = torch.load('./models/single/pretrained-listener-0.pt')
for (file, output_file) in zip(files,output_files):
    metrics, outputs = run(epoch, [file], 'val', 'oracle', speaker, listener, optimizer, loss, vocab, batch_size, True, get_outputs = True)
    preds = outputs['pred'][-1].cpu().numpy()
    scores = outputs['score'][-1].cpu().numpy()
    langs = outputs['lang'][-1].cpu().numpy()
    for game in range(90,langs.shape[0]):
        np.savetxt(output_file+'game_'+str(game)+'_lang.txt', langs[game])
    np.savetxt(output_file+'pred.txt', preds[90:])
    np.savetxt(output_file+'score.txt', scores[90:])
    print(metrics)
    np.save(output_file+'metrics.npy', metrics) 
"""
listener_names = ['train','val','test']
listeners = ['./models/single/pretrained-listener-0.pt','./models/single/pretrained-listener-1.pt','./models/single/pretrained-listener-2.pt'] # or 2, 1
models = ['s','rsa','srr5','s','s_context','pretrained_len']
speakers = ['./models/single/literal-speaker.pt','./models/single/literal-speaker.pt',
            './models/single/literal-speaker.pt','./models/single/literal-speaker.pt',
            './models/single/literal-speaker-contextual.pt','./models/single/pretrained_speaker_len01.pt']
run_types = ['sample','rsa','sample','sample','sample','pragmatic']
activations = ['gumbel','gumbel','gumbel','gumbel','gumbel','gumbel']
num_samples = [1,1,5,1,1,1]
for listener, listener_name in zip(listeners[2:], listener_names[2:]):
    listener = torch.load(listener)
    for i, model, speaker, run_type, activation, n in zip(list(range(6)),models,speakers,run_types,activations,num_samples):
        print(listener_name+' '+model)
        speaker = torch.load(speaker)
        for (file, output_file) in zip(files,output_files):
            print(model, run_type, activation, n)
            metrics, outputs = run(epoch, [file], 'val', run_type, speaker, listener, optimizer, loss, vocab, batch_size, True, num_samples = n, get_outputs = True, activation = activation)
            preds = outputs['pred'][-1].cpu().numpy()
            scores = outputs['score'][-1].cpu().numpy()
            langs = outputs['lang'][-1].cpu().numpy()
            for game in range(langs.shape[0]-10,langs.shape[0]):
                seq = []
                for word_index in langs[game]:
                    seq.append(vocab['i2w'][word_index])
            for game in range(90,langs.shape[0]):
                np.savetxt(output_file+'game_'+str(game)+'_'+str(model)+'_'+str(listener_name)+'_lang.txt', seq, delimiter=" ", fmt="%s")
            np.savetxt(output_file+str(model)+'_'+str(listener_name)+'_pred.txt', preds[90:])
            print(output_file+str(model)+'_'+str(listener_name)+'_pred.txt', preds[90:])
            np.savetxt(output_file+str(model)+'_'+str(listener_name)+'_score.txt', scores[90:])
            print(metrics)
            np.save(output_file+str(model)+'_'+str(listener_name)+'_metrics.npy', metrics) 