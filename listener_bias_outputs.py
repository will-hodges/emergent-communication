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

vocab = torch.load('./models/single/vocab.pt')
print(vocab)

# Optimization
optimizer = None
loss = nn.CrossEntropyLoss()

# Metrics
metrics = init_metrics()

batch_size = 100
files = ['./data/single/random/reference-1000.npz','./data/single/both-needed/reference-1000.npz','./data/single/shape-needed/reference-1000.npz','./data/single/color-needed/reference-1000.npz', './data/single/either-ok/reference-1000.npz']
output_files = ['./output/single/random/','./output/single/both-needed/','./output/single/shape-needed/','./output/single/color-needed/','./output/single/either-ok/']
epoch = 0

print('test color')
speaker = None
listener = torch.load('pretrained-listener-0.pt')
for (file, output_file) in zip(files[1:],output_files[1:]):
    metrics, outputs = run(epoch, [file], 'val', 'test', speaker, listener, optimizer, loss, vocab, batch_size, True, get_outputs = True, test_type = 'color')
    preds = outputs['pred'][-1].cpu().numpy()
    scores = outputs['score'][-1].cpu().numpy()
    langs = outputs['lang'][-1].cpu().numpy()
    for game in range(90,langs.shape[0]):
        np.savetxt(output_file+'game_'+str(game)+'_color_lang.txt', langs[game])
    np.savetxt(output_file+'color_pred.txt', preds[90:])
    np.savetxt(output_file+'color_score.txt', scores[90:])
    print(metrics)
    np.save(output_file+'color_metrics.npy', metrics) 

print('test shape')
speaker = None
listener = torch.load('pretrained-listener-0.pt')
for (file, output_file) in zip(files[1:],output_files[1:]):
    metrics, outputs = run(epoch, [file], 'val', 'test', speaker, listener, optimizer, loss, vocab, batch_size, True, get_outputs = True, test_type = 'shape')
    preds = outputs['pred'][-1].cpu().numpy()
    scores = outputs['score'][-1].cpu().numpy()
    langs = outputs['lang'][-1].cpu().numpy()
    for game in range(90,langs.shape[0]):
        np.savetxt(output_file+'game_'+str(game)+'_shape_lang.txt', langs[game])
    np.savetxt(output_file+'shape_pred.txt', preds[90:])
    np.savetxt(output_file+'shape_score.txt', scores[90:])
    print(metrics)
    np.save(output_file+'shape_metrics.npy', metrics) 

print('test color-shape')
speaker = None
listener = torch.load('pretrained-listener-0.pt')
for (file, output_file) in zip(files[1:],output_files[1:]):
    metrics, outputs = run(epoch, [file], 'val', 'test', speaker, listener, optimizer, loss, vocab, batch_size, True, get_outputs = True, test_type = 'color-shape')
    preds = outputs['pred'][-1].cpu().numpy()
    scores = outputs['score'][-1].cpu().numpy()
    langs = outputs['lang'][-1].cpu().numpy()
    for game in range(90,langs.shape[0]):
        np.savetxt(output_file+'game_'+str(game)+'_color_shape_lang.txt', langs[game])
    np.savetxt(output_file+'color_shape_pred.txt', preds[90:])
    np.savetxt(output_file+'color_shape_score.txt', scores[90:])
    print(metrics)
    np.save(output_file+'color_shape_metrics.npy', metrics) 

print('test shape-color')
speaker = None
listener = torch.load('pretrained-listener-0.pt')
for (file, output_file) in zip(files[1:],output_files[1:]):
    metrics, outputs = run(epoch, [file], 'val', 'test', speaker, listener, optimizer, loss, vocab, batch_size, True, get_outputs = True, test_type = 'shape-color')
    preds = outputs['pred'][-1].cpu().numpy()
    scores = outputs['score'][-1].cpu().numpy()
    langs = outputs['lang'][-1].cpu().numpy()
    for game in range(90,langs.shape[0]):
        np.savetxt(output_file+'game_'+str(game)+'_shape_color_lang.txt', langs[game])
    np.savetxt(output_file+'shape_color_pred.txt', preds[90:])
    np.savetxt(output_file+'shape_color_score.txt', scores[90:])
    print(metrics)
    np.save(output_file+'shape_color_metrics.npy', metrics) 