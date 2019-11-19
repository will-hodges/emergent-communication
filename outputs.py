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

vocab = torch.load('vocab.pt')
print(vocab)

listener = torch.load('pretrained-listener-0.pt')
listener_val = torch.load('pretrained-listener-1.pt')
listener_ci = [torch.load('pretrained-listener-2.pt'),torch.load('pretrained-listener-3.pt'),torch.load('pretrained-listener-4.pt'),torch.load('pretrained-listener-5.pt'),torch.load('pretrained-listener-6.pt'),torch.load('pretrained-listener-7.pt'),torch.load('pretrained-listener-8.pt'),torch.load('pretrained-listener-9.pt'),torch.load('pretrained-listener-10.pt')]

# Optimization
optimizer = None
loss = nn.CrossEntropyLoss()

# Metrics
metrics = init_metrics()

if get_inputs:
    
    directories = ['random','both-needed','color-needed','shape-needed','either-ok']
    for directory in directories:
        d = data.load_raw_data('./data/single/'+str(directory)+'/reference-1000.npz')
        dataloader = DataLoader(ShapeWorld(d, vocab), batch_size=1, shuffle=False)
        for game, (img, y, lang) in enumerate(dataloader):
            if game > (989):
                plt.imsave('./output/'+str(directory)+'/game-'+str(game-990)+'-img-1.png', img[0][0].permute(1,2,0).cpu().numpy())
                plt.imsave('./output/'+str(directory)+'/game-'+str(game-990)+'-img-2.png', img[0][1].permute(1,2,0).cpu().numpy())
                plt.imsave('./output/'+str(directory)+'/game-'+str(game-990)+'-img-3.png', img[0][2].permute(1,2,0).cpu().numpy())
                np.savetxt('./output/'+str(directory)+'/game-'+str(game-990)+'-y.txt', y)
                np.savetxt('./output/'+str(directory)+'/game-'+str(game-990)+'-lang.txt', lang)
                
else:        
    
    batch_size = 100
    files = ['./data/single/random/reference-1000.npz','./data/single/both-needed/reference-1000.npz','./data/single/shape-needed/reference-1000.npz','./data/single/color-needed/reference-1000.npz', './data/single/either-ok/reference-1000.npz']
    output_files = ['./output/random/','./output/both-needed/','./output/shape-needed/','./output/color-needed/','./output/either-ok/']
    epoch = 0

    print('sample val')
    speaker = torch.load('literal-speaker.pt')
    listener = torch.load('pretrained-listener-1.pt')
    for (file, output_file) in zip(files,output_files):
        metrics, outputs = run(epoch, [file], 'val', 'sample', speaker, listener, optimizer, loss, vocab, batch_size, True, num_samples = 1, get_outputs = True)
        preds = outputs['pred'][-1].cpu().numpy()
        scores = outputs['score'][-1].cpu().numpy()
        langs = outputs['lang'][-1].cpu().numpy()
        for game in range(90,langs.shape[0]):
            np.savetxt(output_file+'game_'+str(game)+'_s_val_lang.txt', langs[game])
        np.savetxt(output_file+'s_val_pred.txt', preds[90:])
        np.savetxt(output_file+'s_val_score.txt', scores[90:])
        print(metrics)
        np.save(output_file+'s_val_metrics.npy', metrics) 

    print('sample/rerank (5) val')
    speaker = torch.load('literal-speaker.pt')
    listener = torch.load('pretrained-listener-0.pt')
    for (file, output_file) in zip(files,output_files):
        metrics, outputs = run(epoch, [file], 'val', 'sample', speaker, listener, optimizer, loss, vocab, batch_size, True, num_samples = 5, get_outputs = True)
        preds = outputs['pred'][-1].cpu().numpy()
        scores = outputs['score'][-1].cpu().numpy()
        langs = outputs['lang'][-1].cpu().numpy()
        for game in range(90,langs.shape[0]):
            np.savetxt(output_file+'game_'+str(game)+'_srr5_val_lang.txt', langs[game])
        np.savetxt(output_file+'srr5_val_pred.txt', preds[90:])
        np.savetxt(output_file+'srr5_val_score.txt', scores[90:])
        print(metrics)
        np.save(output_file+'srr5_val_metrics.npy', metrics) 
    print('sample/rerank (10) val')
    speaker = torch.load('literal-speaker.pt')
    listener = torch.load('pretrained-listener-0.pt')
    for (file, output_file) in zip(files,output_files):
        metrics, outputs = run(epoch, [file], 'val', 'sample', speaker, listener, optimizer, loss, vocab, batch_size, True, num_samples = 10, get_outputs = True)
        preds = outputs['pred'][-1].cpu().numpy()
        scores = outputs['score'][-1].cpu().numpy()
        langs = outputs['lang'][-1].cpu().numpy()
        for game in range(90,langs.shape[0]):
            np.savetxt(output_file+'game_'+str(game)+'_srr10_val_lang.txt', langs[game])
        np.savetxt(output_file+'srr10_val_pred.txt', preds[90:])
        np.savetxt(output_file+'srr10_val_score.txt', scores[90:])
        print(metrics)
        np.save(output_file+'srr10_val_metrics.npy', metrics) 
    print('sample/rerank (20) val')
    speaker = torch.load('literal-speaker.pt')
    listener = torch.load('pretrained-listener-0.pt')
    for (file, output_file) in zip(files,output_files):
        metrics, outputs = run(epoch, [file], 'val', 'sample', speaker, listener, optimizer, loss, vocab, batch_size, True, num_samples = 20, get_outputs = True)
        preds = outputs['pred'][-1].cpu().numpy()
        scores = outputs['score'][-1].cpu().numpy()
        langs = outputs['lang'][-1].cpu().numpy()
        for game in range(90,langs.shape[0]):
            np.savetxt(output_file+'game_'+str(game)+'_srr20_val_lang.txt', langs[game])
        np.savetxt(output_file+'srr20_val_pred.txt', preds[90:])
        np.savetxt(output_file+'srr20_val_score.txt', scores[90:])
        print(metrics)
        np.save(output_file+'srr20_val_metrics.npy', metrics) 

    print('pretrained val')
    speaker = torch.load('pretrained-speaker.pt')
    listener = torch.load('pretrained-listener-1.pt')
    for (file, output_file) in zip(files,output_files):
        metrics, outputs = run(epoch, [file], 'val', 'pragmatic', speaker, listener, optimizer, loss, vocab, batch_size, True, get_outputs = True)
        preds = outputs['pred'][-1].cpu().numpy()
        scores = outputs['score'][-1].cpu().numpy()
        langs = outputs['lang'][-1].cpu().numpy()
        for game in range(90,langs.shape[0]):
            np.savetxt(output_file+'game_'+str(game)+'_pretrained_val_lang.txt', langs[game])
        np.savetxt(output_file+'pretrained_val_pred.txt', preds[90:])
        np.savetxt(output_file+'pretrained_val_score.txt', scores[90:])
        print(metrics)
        np.save(output_file+'pretrained_val_metrics.npy', metrics) 

    print('sample train')
    speaker = torch.load('literal-speaker.pt')
    listener = torch.load('pretrained-listener-0.pt')
    for (file, output_file) in zip(files,output_files):
        metrics, outputs = run(epoch, [file], 'val', 'sample', speaker, listener, optimizer, loss, vocab, batch_size, True, num_samples = 1, get_outputs = True)
        preds = outputs['pred'][-1].cpu().numpy()
        scores = outputs['score'][-1].cpu().numpy()
        langs = outputs['lang'][-1].cpu().numpy()
        for game in range(90,langs.shape[0]):
            np.savetxt(output_file+'game_'+str(game)+'_s_train_lang.txt', langs[game])
        np.savetxt(output_file+'s_train_pred.txt', preds[90:])
        np.savetxt(output_file+'s_train_score.txt', scores[90:])
        print(metrics)
        np.save(output_file+'s_train_metrics.npy', metrics) 

    print('sample/rerank (5) train')
    speaker = torch.load('literal-speaker.pt')
    listener = torch.load('pretrained-listener-01.pt')
    for (file, output_file) in zip(files,output_files):
        metrics, outputs = run(epoch, [file], 'val', 'sample', speaker, listener, optimizer, loss, vocab, batch_size, True, num_samples = 5, get_outputs = True)
        preds = outputs['pred'][-1].cpu().numpy()
        scores = outputs['score'][-1].cpu().numpy()
        langs = outputs['lang'][-1].cpu().numpy()
        for game in range(90,langs.shape[0]):
            np.savetxt(output_file+'game_'+str(game)+'_srr5_train_lang.txt', langs[game])
        np.savetxt(output_file+'srr5_train_pred.txt', preds[90:])
        np.savetxt(output_file+'srr5_train_score.txt', scores[90:])
        print(metrics)
        np.save(output_file+'srr5_train_metrics.npy', metrics)
    print('sample/rerank (10) train')
    speaker = torch.load('literal-speaker.pt')
    listener = torch.load('pretrained-listener-01.pt')
    for (file, output_file) in zip(files,output_files):
        metrics, outputs = run(epoch, [file], 'val', 'sample', speaker, listener, optimizer, loss, vocab, batch_size, True, num_samples = 10, get_outputs = True)
        preds = outputs['pred'][-1].cpu().numpy()
        scores = outputs['score'][-1].cpu().numpy()
        langs = outputs['lang'][-1].cpu().numpy()
        for game in range(90,langs.shape[0]):
            np.savetxt(output_file+'game_'+str(game)+'_srr10_train_lang.txt', langs[game])
        np.savetxt(output_file+'srr10_train_pred.txt', preds[90:])
        np.savetxt(output_file+'srr10_train_score.txt', scores[90:])
        print(metrics)
        np.save(output_file+'srr10_train_metrics.npy', metrics) 
    print('sample/rerank (20) train')
    speaker = torch.load('literal-speaker.pt')
    listener = torch.load('pretrained-listener-01.pt')
    for (file, output_file) in zip(files,output_files):
        metrics, outputs = run(epoch, [file], 'val', 'sample', speaker, listener, optimizer, loss, vocab, batch_size, True, num_samples = 20, get_outputs = True)
        preds = outputs['pred'][-1].cpu().numpy()
        scores = outputs['score'][-1].cpu().numpy()
        langs = outputs['lang'][-1].cpu().numpy()
        for game in range(90,langs.shape[0]):
            np.savetxt(output_file+'game_'+str(game)+'_srr20_train_lang.txt', langs[game])
        np.savetxt(output_file+'srr20_train_pred.txt', preds[90:])
        np.savetxt(output_file+'srr20_train_score.txt', scores[90:])
        print(metrics)
        np.save(output_file+'srr20_train_metrics.npy', metrics) 

    print('pretrained train')
    speaker = torch.load('pretrained-speaker.pt')
    listener = torch.load('pretrained-listener-0.pt')
    for (file, output_file) in zip(files,output_files):
        metrics, outputs = run(epoch, [file], 'val', 'pragmatic', speaker, listener, optimizer, loss, vocab, batch_size, True, get_outputs = True)
        scores = outputs['score'][-1].cpu().numpy()
        preds = outputs['pred'][-1].cpu().numpy()
        langs = outputs['lang'][-1].cpu().numpy()
        for game in range(90,langs.shape[0]):
            np.savetxt(output_file+'game_'+str(game)+'_pretrained_train_lang.txt', langs[game])
        np.savetxt(output_file+'pretrained_train_pred.txt', preds[90:])
        np.savetxt(output_file+'pretrained_train_score.txt', scores[90:])
        print(metrics)
        np.save(output_file+'pretrained_train_metrics.npy', metrics) 