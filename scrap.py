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

def compute_average_metrics(meters):
    """
    Compute averages from meters. Handle tensors vs floats (always return a
    float)

    Parameters
    ----------
    meters : Dict[str, util.AverageMeter]
        Dict of average meters, whose averages may be of type ``float`` or ``torch.Tensor``

    Returns
    -------
    metrics : Dict[str, float]
        Average value of each metric
    """
    metrics = {m: vs.avg for m, vs in meters.items()}
    metrics = {
        m: v if isinstance(v, float) else v.item()
        for m, v in metrics.items()
    }
    return metrics

VOCAB = ['gray', 'shape', 'blue', 'square', 'circle', 'green', 'red', 'rectangle', 'yellow', 'ellipse', 'white']

# Vocab
speaker_embs = nn.Embedding(4 + len(VOCAB), 50)
listener_embs = nn.Embedding(4 + len(VOCAB), 50)
vocab = torch.load('single_vocab.pt')

listener = torch.load('pretrained_listener2_small.pt')
#listener = torch.load('pretrained_listener.pt')
speaker_vision = vision.Conv4()
speaker = models.Speaker(speaker_vision, speaker_embs, 'reference')
speaker = speaker.cuda()
listener = listener.cuda()

# Optimization
optimizer = optim.Adam(list(speaker.parameters())+list(listener.parameters()),lr=1e-3)
loss = nn.CrossEntropyLoss()

data_file = ['./data/single/reference-1000-4.npz']

listener.eval()
context = torch.no_grad()  # Do not evaluate gradients for efficiency

# Initialize your average meters to keep track of the epoch's running average
measures = ['loss', 'acc']
meters = {m: util.AverageMeter() for m in measures}

with context:
    for file in data_file:
        d = data.load_raw_data(file)
        dataloader = DataLoader(ShapeWorld(d, vocab), batch_size=32, shuffle=True)
        for batch_i, (img, y, lang) in enumerate(dataloader):
                    
            y = y.argmax(1) # convert from onehot
            batch_size = img.shape[0]

            # Convert to float
            img = img.float()

            max_len = 20
            length = torch.tensor([np.count_nonzero(t) for t in lang.cpu()], dtype=np.int)
            lang = F.one_hot(lang, num_classes = 4+len(VOCAB))
            lang = F.pad(lang,(0,0,0,max_len-lang.shape[1])).float()
            for B in range(lang.shape[0]):
                for L in range(lang.shape[1]):
                    if lang[B][L].sum() == 0:
                        lang[B][L][0] = 1

            img = img.cuda()
            y = y.cuda()
            lang.cuda()
            length = length.cuda()

            # Forward pass
            lis_scores = listener(img, lang, length)

            # Evaluate loss and accuracy
            this_loss = loss(lis_scores, y)
            lis_pred = lis_scores.argmax(1)
            this_acc = (lis_pred == y).float().mean().item()

            meters['loss'].update(this_loss, batch_size)
            meters['acc'].update(this_acc, batch_size)

    metrics = compute_average_metrics(meters)
    print(metrics)