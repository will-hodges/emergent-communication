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

vocab = torch.load('./models/single/vocab.pt')
print(vocab)

# Optimization
optimizer = None
loss = nn.CrossEntropyLoss()

# Metrics
metrics = init_metrics()
   
batch_size = 100

file_dict = {'new': ['./data/single/generalization_new_combo/test/reference-1000.npz','./data/single/generalization_new_context/test/reference-1000.npz','./data/single/generalization_new_color/test/reference-1000.npz','./data/single/generalization_new_shape/test/reference-1000.npz'], 
             'train': ['./data/single/generalization_new_combo/reference-1000-0.npz','./data/single/generalization_new_context/reference-1000-0.npz','./data/single/generalization_new_color/reference-1000-0.npz','./data/single/generalization_new_shape/reference-1000-0.npz']}
output_files = ['./output/single/generalization_new_combo/','./output/single/generalization_new_context/','./output/single/generalization_new_color/','./output/single/generalization_new_shape/']
epoch = 0

listener_names = ['train','val','test']
listeners = ['./models/single/pretrained-listener-0.pt','./models/single/pretrained-listener-1.pt','./models/single/pretrained-listener-2.pt']
speaker_dirs = ['./models/single/new_combo_','./models/single/new_context_','./models/single/new_color_','./models/single/new_shape_']

models = ['s_context','s','srr5','rsa','pretrained_len']
speakers = ['literal_speaker_contextual.pt','literal_speaker.pt',
            'literal_speaker.pt','literal_speaker.pt','pretrained_speaker.pt']
run_types = ['sample','sample','sample','rsa','pragmatic','pragmatic']
activations = ['gumbel','gumbel','gumbel','gumbel','gumbel','gumbel']
num_samples = [1,1,5,10,20,1,1,1]

for ts, files in file_dict.items():
    for listener, listener_name in zip(listeners, listener_names):
        count = 0
        listener = torch.load(listener)
        for speaker_dir, file, output_file in zip(speaker_dirs,files,output_files):
            for model, speaker, run_type, activation, n in zip(models,speakers,run_types,activations,num_samples):
                speaker = torch.load(speaker_dir+speaker).cuda()
                metrics, outputs = run(0, [file], 'val', run_type, speaker, listener, optimizer, loss, vocab, batch_size, True, num_samples = n, get_outputs = True, activation = activation)
                preds = outputs['pred'][-1].cpu().numpy()
                scores = outputs['score'][-1].cpu().numpy()
                langs = outputs['lang'][-1].cpu().numpy()
                for game in range(90,langs.shape[0]):
                    np.savetxt(output_file+'game_'+str(game)+'_'+str(model)+'_'+str(listener_name)+'_lang.txt', langs[game])
                np.savetxt(output_file+str(model)+'_'+str(listener_name)+'_pred.txt', preds[90:])
                np.savetxt(output_file+str(model)+'_'+str(listener_name)+'_score.txt', scores[90:])
                print(metrics['acc'])
                np.save(output_file+str(model)+'_'+str(listener_name)+'_metrics.npy', metrics) 
