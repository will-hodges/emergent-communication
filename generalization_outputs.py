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

files = ['./data/single/generalization_new_combo/test/reference-1000.npz','./data/single/generalization_new_context/test/reference-1000.npz','./data/single/generalization_new_color/test/reference-1000.npz','./data/single/generalization_new_shape/test/reference-1000.npz']
output_files = ['./output/single/generalization_new_combo/','./output/single/generalization_new_context/','./output/single/generalization_new_color/','./output/single/generalization_new_shape/']
epoch = 0

directories = ['generalization_new_combo','generalization_new_context','generalization_new_color','generalization_new_shape']
for directory in directories:
    d = data.load_raw_data('./data/single/'+str(directory)+'/test/reference-1000.npz')
    dataloader = DataLoader(ShapeWorld(d, vocab), batch_size=1, shuffle=False)
    for game, (img, y, lang) in enumerate(dataloader):
        if game > (989):
            plt.imsave('./output/single/'+str(directory)+'/game-'+str(game-990)+'-img-1.png', img[0][0].permute(1,2,0).cpu().numpy())
            plt.imsave('./output/single/'+str(directory)+'/game-'+str(game-990)+'-img-2.png', img[0][1].permute(1,2,0).cpu().numpy())
            plt.imsave('./output/single/'+str(directory)+'/game-'+str(game-990)+'-img-3.png', img[0][2].permute(1,2,0).cpu().numpy())
            np.savetxt('./output/single/'+str(directory)+'/game-'+str(game-990)+'-y.txt', y)
            np.savetxt('./output/single/'+str(directory)+'/game-'+str(game-990)+'-lang.txt', lang)


listener_names = ['train','test','val']
listeners = ['./models/single/pretrained-listener-0.pt','./models/single/pretrained-listener-1.pt','./models/single/pretrained-listener-2.pt']
models = ['new_combo','new_context','new_color','new_shape']
speakers = ['./models/single/new_combo_pretrained_speaker.pt','./models/single/new_context_pretrained_speaker.pt','./models/single/new_color_pretrained_speaker.pt','./models/single/new_shape_pretrained_speaker.pt']
run_types = ['pragmatic','pragmatic','pragmatic','pragmatic']
num_samples = [1,5,10,20]
n = 1
for listener, listener_name in zip(listeners, listener_names):
    count = 0
    listener = torch.load(listener)
    for model, speaker, run_type, file, output_file in zip(models,speakers,run_types,files,output_files):
        print(listener_name+' '+model)
        if run_type == 'sample':
            n = num_samples[count]
            count += 1
        speaker = torch.load(speaker)
        metrics, outputs = run(epoch, [file], 'val', run_type, speaker, listener, optimizer, loss, vocab, batch_size, True, num_samples = n, get_outputs = True)
        preds = outputs['pred'][-1].cpu().numpy()
        scores = outputs['score'][-1].cpu().numpy()
        langs = outputs['lang'][-1].cpu().numpy()
        for game in range(90,langs.shape[0]):
            np.savetxt(output_file+'game_'+str(game)+'_'+str(model)+'_'+str(listener_name)+'_lang.txt', langs[game])
        np.savetxt(output_file+str(model)+'_'+str(listener_name)+'_pred.txt', preds[90:])
        print(output_file+str(model)+'_'+str(listener_name)+'_pred.txt', preds[90:])
        np.savetxt(output_file+str(model)+'_'+str(listener_name)+'_score.txt', scores[90:])
        print(metrics)
        np.save(output_file+str(model)+'_'+str(listener_name)+'_metrics.npy', metrics) 