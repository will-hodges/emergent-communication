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

get_inputs = False

# Optimization
optimizer = None
loss = nn.CrossEntropyLoss()

# Metrics
metrics = init_metrics()
   
batch_size = 100
files = ['./data/single/random/reference-1000.npz','./data/chairs/data_1000_83.npz','./data/colors/data_1000_45.npz']
output_files = ['./output/single/','./output/chairs/','./output/colors/']
directories = ['single','chairs','colors']
listener_names = ['train','val','test']
listeners = ['pretrained-listener-0.pt','pretrained-listener-1.pt','pretrained-listener-2.pt']
models = ['s','srr5','srr10','srr20','human','direct']
speakers = ['literal-speaker.pt','literal-speaker.pt',
            'literal-speaker.pt','literal-speaker.pt','reg','len']
run_types = ['sample','sample','sample','sample','pragmatic','pragmatic']
num_samples = [1,5,10,20,1,1]
for (file, output_file, directory) in zip(files[2:],output_files[2:],directories[2:]):
    vocab = torch.load('./models/'+directory+'/vocab.pt')
    for listener, listener_name in zip(listeners, listener_names):
        count = 0
        listener = torch.load('./models/'+directory+'/'+listener)
        for model, speaker, run_type, n in zip(models,speakers,run_types,num_samples):
            if speaker == 'reg':
                if directory == 'chairs':
                    speaker = 'pretrained_speaker_softmax.pt'
                if directory == 'colors':
                    speaker = 'pretrained_speaker.pt'
            if speaker == 'len':
                if directory == 'chairs':
                    speaker = 'pretrained_speaker_len_001_softmax.pt'
                if directory == 'colors':
                    speaker = 'pretrained_speaker_softmax.pt'
            print(listener_name+' '+model+' '+directory)
            speaker = torch.load('./models/'+directory+'/'+speaker)
            metrics, outputs = run(0, [file], 'val', run_type, speaker, listener, optimizer, loss, vocab, batch_size, True, num_samples = n, get_outputs = True, activation = 'gumbel', ci = False, dataset = directory)
            preds = outputs['pred'][-1].cpu().numpy()
            scores = outputs['score'][-1].cpu().numpy()
            langs = outputs['lang'][-1].cpu().numpy()
            for game in range(90,langs.shape[0]):
                seq = []
                for word_index in langs[game]:
                    seq.append(vocab['i2w'][word_index])
                print(seq)
                np.savetxt(output_file+'game_'+str(game)+'_'+str(model)+'_'+str(listener_name)+'_lang.txt', seq, delimiter=" ", fmt="%s")
            np.savetxt(output_file+str(model)+'_'+str(listener_name)+'_pred.txt', preds[90:])
            print(output_file+str(model)+'_'+str(listener_name)+'_pred.txt', preds[90:])
            np.savetxt(output_file+str(model)+'_'+str(listener_name)+'_score.txt', scores[90:])
            print(metrics)
            np.save(output_file+str(model)+'_'+str(listener_name)+'_metrics.npy', metrics) 