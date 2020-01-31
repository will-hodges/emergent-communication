"""
Train an RNN decoder to make binary predictions;
then train an RNN language model to generate sequences
"""

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
import pandas as pd

import vision
import util
from data import ShapeWorld
import data

from chairs import ChairsInContext
from colors import ColorsInContext

from run import run
import models

def init_metrics():
    """
    Initialize the metrics for this training run. This is a defaultdict, so
    metrics not specified here can just be appended to/assigned to during
    training.
    Returns
    -------
    metrics : `collections.defaultdict`
        All training metrics
    """
    metrics = defaultdict(list)
    metrics['best_acc'] = 0.0
    metrics['best_loss'] = float('inf')
    metrics['best_epoch'] = 0
    return metrics

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(description='Train', formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--new', action='store_true')
    parser.add_argument('--dataset', default='single')
    parser.add_argument('--game_type', choices=['concept', 'reference'], default='reference', type=str)
    parser.add_argument('--srr', action='store_true')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--new_vocab', action='store_true')
    parser.add_argument('--lm_wt', default=0.1)
    parser.add_argument('--metrics_file', default='metrics.csv', help='Where to save metrics')
    parser.add_argument('--generalization', default=None)
    parser.add_argument('--activation', default=None)
    parser.add_argument('--penalty', default=None)
    parser.add_argument('--tau', default=1, type=float)
    args = parser.parse_args()
    
    print('/pretrained_speaker_'+str(int(args.tau*100))+'.pt')
    
    # Data
    if args.dataset == 'single':
        pretrain_data = [['./data/single/reference-1000-0.npz','./data/single/reference-1000-1.npz','./data/single/reference-1000-2.npz','./data/single/reference-1000-3.npz','./data/single/reference-1000-4.npz'],
                                       ['./data/single/reference-1000-5.npz','./data/single/reference-1000-6.npz','./data/single/reference-1000-7.npz','./data/single/reference-1000-8.npz','./data/single/reference-1000-9.npz'],
                                       ['./data/single/reference-1000-10.npz','./data/single/reference-1000-11.npz','./data/single/reference-1000-12.npz','./data/single/reference-1000-13.npz','./data/single/reference-1000-14.npz'],
                                       ['./data/single/reference-1000-15.npz','./data/single/reference-1000-16.npz','./data/single/reference-1000-17.npz','./data/single/reference-1000-18.npz','./data/single/reference-1000-19.npz'],
                                       ['./data/single/reference-1000-20.npz','./data/single/reference-1000-21.npz','./data/single/reference-1000-22.npz','./data/single/reference-1000-23.npz','./data/single/reference-1000-24.npz'],
                                       ['./data/single/reference-1000-25.npz','./data/single/reference-1000-26.npz','./data/single/reference-1000-27.npz','./data/single/reference-1000-28.npz','./data/single/reference-1000-29.npz'],
                                       ['./data/single/reference-1000-30.npz','./data/single/reference-1000-31.npz','./data/single/reference-1000-32.npz','./data/single/reference-1000-33.npz','./data/single/reference-1000-34.npz'],
                                       ['./data/single/reference-1000-35.npz','./data/single/reference-1000-36.npz','./data/single/reference-1000-37.npz','./data/single/reference-1000-38.npz','./data/single/reference-1000-39.npz'],
                                       ['./data/single/reference-1000-40.npz','./data/single/reference-1000-41.npz','./data/single/reference-1000-42.npz','./data/single/reference-1000-43.npz','./data/single/reference-1000-44.npz'],
                                       ['./data/single/reference-1000-45.npz','./data/single/reference-1000-46.npz','./data/single/reference-1000-47.npz','./data/single/reference-1000-48.npz','./data/single/reference-1000-49.npz'],
                                       ['./data/single/reference-1000-50.npz','./data/single/reference-1000-51.npz','./data/single/reference-1000-52.npz','./data/single/reference-1000-53.npz','./data/single/reference-1000-54.npz'],
                                       ['./data/single/reference-1000-70.npz','./data/single/reference-1000-71.npz','./data/single/reference-1000-72.npz','./data/single/reference-1000-73.npz','./data/single/reference-1000-74.npz']]

        if args.generalization != None:
            train_data = ['./data/single/generalization_'+args.generalization+'/reference-1000-60.npz','./data/single/generalization_'+args.generalization+'/reference-1000-61.npz','./data/single/generalization_'+args.generalization+'/reference-1000-62.npz','./data/single/generalization_'+args.generalization+'/reference-1000-63.npz','./data/single/generalization_'+args.generalization+'/reference-1000-64.npz']
            val_data = ['./data/single/generalization_'+args.generalization+'/reference-1000-65.npz','./data/single/generalization_'+args.generalization+'/reference-1000-66.npz','./data/single/generalization_'+args.generalization+'/reference-1000-67.npz','./data/single/generalization_'+args.generalization+'/reference-1000-68.npz','./data/single/generalization_'+args.generalization+'/reference-1000-69.npz']
        else: 
            train_data = ['./data/single/reference-1000-60.npz','./data/single/reference-1000-61.npz','./data/single/reference-1000-62.npz','./data/single/reference-1000-63.npz','./data/single/reference-1000-64.npz']
            val_data = ['./data/single/reference-1000-65.npz','./data/single/reference-1000-66.npz','./data/single/reference-1000-67.npz','./data/single/reference-1000-68.npz','./data/single/reference-1000-69.npz']
    else:
        if args.dataset == 'chairs':
            DatasetClass = ChairsInContext
        else:
            DatasetClass = ColorsInContext
        
        if args.pretrain or args.srr:
            if args.dataset == 'chairs':
                pretrain_data = [['./data/'+args.dataset+'/data_1000_0.npz','./data/'+args.dataset+'/data_1000_1.npz','./data/'+args.dataset+'/data_1000_2.npz','./data/'+args.dataset+'/data_1000_3.npz','./data/'+args.dataset+'/data_1000_4.npz','./data/'+args.dataset+'/data_1000_5.npz','./data/'+args.dataset+'/data_1000_6.npz','./data/'+args.dataset+'/data_1000_7.npz','./data/'+args.dataset+'/data_1000_8.npz','./data/'+args.dataset+'/data_1000_9.npz','./data/'+args.dataset+'/data_1000_10.npz','./data/'+args.dataset+'/data_1000_11.npz','./data/'+args.dataset+'/data_1000_12.npz','./data/'+args.dataset+'/data_1000_13.npz','./data/'+args.dataset+'/data_1000_14.npz','./data/'+args.dataset+'/data_1000_15.npz','./data/'+args.dataset+'/data_1000_16.npz','./data/'+args.dataset+'/data_1000_17.npz','./data/'+args.dataset+'/data_1000_18.npz','./data/'+args.dataset+'/data_1000_19.npz','./data/'+args.dataset+'/data_1000_20.npz','./data/'+args.dataset+'/data_1000_21.npz','./data/'+args.dataset+'/data_1000_22.npz','./data/'+args.dataset+'/data_1000_23.npz','./data/'+args.dataset+'/data_1000_24.npz','./data/'+args.dataset+'/data_1000_25.npz','./data/'+args.dataset+'/data_1000_26.npz'],
                                 ['./data/'+args.dataset+'/data_1000_27.npz','./data/'+args.dataset+'/data_1000_28.npz','./data/'+args.dataset+'/data_1000_29.npz','./data/'+args.dataset+'/data_1000_30.npz','./data/'+args.dataset+'/data_1000_31.npz','./data/'+args.dataset+'/data_1000_32.npz','./data/'+args.dataset+'/data_1000_33.npz','./data/'+args.dataset+'/data_1000_34.npz','./data/'+args.dataset+'/data_1000_35.npz','./data/'+args.dataset+'/data_1000_36.npz','./data/'+args.dataset+'/data_1000_37.npz','./data/'+args.dataset+'/data_1000_38.npz','./data/'+args.dataset+'/data_1000_39.npz','./data/'+args.dataset+'/data_1000_40.npz','./data/'+args.dataset+'/data_1000_41.npz','./data/'+args.dataset+'/data_1000_42.npz','./data/'+args.dataset+'/data_1000_43.npz','./data/'+args.dataset+'/data_1000_44.npz','./data/'+args.dataset+'/data_1000_45.npz','./data/'+args.dataset+'/data_1000_46.npz','./data/'+args.dataset+'/data_1000_47.npz','./data/'+args.dataset+'/data_1000_48.npz','./data/'+args.dataset+'/data_1000_49.npz','./data/'+args.dataset+'/data_1000_50.npz','./data/'+args.dataset+'/data_1000_51.npz','./data/'+args.dataset+'/data_1000_52.npz','./data/'+args.dataset+'/data_1000_53.npz'],
                                 ['./data/'+args.dataset+'/data_1000_54.npz','./data/'+args.dataset+'/data_1000_55.npz','./data/'+args.dataset+'/data_1000_56.npz','./data/'+args.dataset+'/data_1000_57.npz','./data/'+args.dataset+'/data_1000_58.npz','./data/'+args.dataset+'/data_1000_59.npz','./data/'+args.dataset+'/data_1000_60.npz','./data/'+args.dataset+'/data_1000_61.npz','./data/'+args.dataset+'/data_1000_62.npz','./data/'+args.dataset+'/data_1000_63.npz','./data/'+args.dataset+'/data_1000_64.npz','./data/'+args.dataset+'/data_1000_65.npz','./data/'+args.dataset+'/data_1000_66.npz','./data/'+args.dataset+'/data_1000_67.npz','./data/'+args.dataset+'/data_1000_68.npz','./data/'+args.dataset+'/data_1000_69.npz','./data/'+args.dataset+'/data_1000_70.npz','./data/'+args.dataset+'/data_1000_71.npz','./data/'+args.dataset+'/data_1000_72.npz','./data/'+args.dataset+'/data_1000_73.npz','./data/'+args.dataset+'/data_1000_74.npz','./data/'+args.dataset+'/data_1000_75.npz','./data/'+args.dataset+'/data_1000_76.npz','./data/'+args.dataset+'/data_1000_77.npz','./data/'+args.dataset+'/data_1000_78.npz','./data/'+args.dataset+'/data_1000_79.npz','./data/'+args.dataset+'/data_1000_80.npz']]
                """
                train_data = ['./data/'+args.dataset+'/data_1000_0.npz','./data/'+args.dataset+'/data_1000_1.npz','./data/'+args.dataset+'/data_1000_2.npz','./data/'+args.dataset+'/data_1000_3.npz','./data/'+args.dataset+'/data_1000_4.npz','./data/'+args.dataset+'/data_1000_5.npz','./data/'+args.dataset+'/data_1000_6.npz','./data/'+args.dataset+'/data_1000_7.npz','./data/'+args.dataset+'/data_1000_8.npz','./data/'+args.dataset+'/data_1000_9.npz','./data/'+args.dataset+'/data_1000_10.npz','./data/'+args.dataset+'/data_1000_11.npz','./data/'+args.dataset+'/data_1000_12.npz','./data/'+args.dataset+'/data_1000_13.npz','./data/'+args.dataset+'/data_1000_14.npz','./data/'+args.dataset+'/data_1000_15.npz','./data/'+args.dataset+'/data_1000_16.npz','./data/'+args.dataset+'/data_1000_17.npz','./data/'+args.dataset+'/data_1000_18.npz','./data/'+args.dataset+'/data_1000_19.npz','./data/'+args.dataset+'/data_1000_20.npz','./data/'+args.dataset+'/data_1000_21.npz','./data/'+args.dataset+'/data_1000_22.npz','./data/'+args.dataset+'/data_1000_23.npz','./data/'+args.dataset+'/data_1000_24.npz','./data/'+args.dataset+'/data_1000_25.npz','./data/'+args.dataset+'/data_1000_26.npz']
                val_data = ['./data/'+args.dataset+'/data_1000_27.npz','./data/'+args.dataset+'/data_1000_28.npz','./data/'+args.dataset+'/data_1000_29.npz','./data/'+args.dataset+'/data_1000_30.npz','./data/'+args.dataset+'/data_1000_31.npz','./data/'+args.dataset+'/data_1000_32.npz','./data/'+args.dataset+'/data_1000_33.npz','./data/'+args.dataset+'/data_1000_34.npz','./data/'+args.dataset+'/data_1000_35.npz','./data/'+args.dataset+'/data_1000_36.npz','./data/'+args.dataset+'/data_1000_37.npz','./data/'+args.dataset+'/data_1000_38.npz','./data/'+args.dataset+'/data_1000_39.npz','./data/'+args.dataset+'/data_1000_40.npz','./data/'+args.dataset+'/data_1000_41.npz','./data/'+args.dataset+'/data_1000_42.npz','./data/'+args.dataset+'/data_1000_43.npz','./data/'+args.dataset+'/data_1000_44.npz','./data/'+args.dataset+'/data_1000_45.npz','./data/'+args.dataset+'/data_1000_46.npz','./data/'+args.dataset+'/data_1000_47.npz','./data/'+args.dataset+'/data_1000_48.npz','./data/'+args.dataset+'/data_1000_49.npz','./data/'+args.dataset+'/data_1000_50.npz','./data/'+args.dataset+'/data_1000_51.npz','./data/'+args.dataset+'/data_1000_52.npz','./data/'+args.dataset+'/data_1000_53.npz']
                """
                train_data = ['./data/'+args.dataset+'/data_1000_0.npz','./data/'+args.dataset+'/data_1000_1.npz','./data/'+args.dataset+'/data_1000_2.npz','./data/'+args.dataset+'/data_1000_3.npz','./data/'+args.dataset+'/data_1000_4.npz','./data/'+args.dataset+'/data_1000_5.npz','./data/'+args.dataset+'/data_1000_6.npz','./data/'+args.dataset+'/data_1000_7.npz','./data/'+args.dataset+'/data_1000_8.npz','./data/'+args.dataset+'/data_1000_9.npz','./data/'+args.dataset+'/data_1000_10.npz','./data/'+args.dataset+'/data_1000_11.npz','./data/'+args.dataset+'/data_1000_12.npz','./data/'+args.dataset+'/data_1000_13.npz','./data/'+args.dataset+'/data_1000_14.npz','./data/'+args.dataset+'/data_1000_15.npz','./data/'+args.dataset+'/data_1000_16.npz','./data/'+args.dataset+'/data_1000_17.npz','./data/'+args.dataset+'/data_1000_18.npz','./data/'+args.dataset+'/data_1000_19.npz','./data/'+args.dataset+'/data_1000_20.npz','./data/'+args.dataset+'/data_1000_21.npz','./data/'+args.dataset+'/data_1000_22.npz','./data/'+args.dataset+'/data_1000_23.npz','./data/'+args.dataset+'/data_1000_24.npz','./data/'+args.dataset+'/data_1000_25.npz','./data/'+args.dataset+'/data_1000_26.npz','./data/'+args.dataset+'/data_1000_27.npz','./data/'+args.dataset+'/data_1000_28.npz','./data/'+args.dataset+'/data_1000_29.npz','./data/'+args.dataset+'/data_1000_30.npz','./data/'+args.dataset+'/data_1000_31.npz','./data/'+args.dataset+'/data_1000_32.npz','./data/'+args.dataset+'/data_1000_33.npz','./data/'+args.dataset+'/data_1000_34.npz','./data/'+args.dataset+'/data_1000_35.npz','./data/'+args.dataset+'/data_1000_36.npz','./data/'+args.dataset+'/data_1000_37.npz','./data/'+args.dataset+'/data_1000_38.npz','./data/'+args.dataset+'/data_1000_39.npz','./data/'+args.dataset+'/data_1000_40.npz','./data/'+args.dataset+'/data_1000_41.npz','./data/'+args.dataset+'/data_1000_42.npz','./data/'+args.dataset+'/data_1000_43.npz','./data/'+args.dataset+'/data_1000_44.npz','./data/'+args.dataset+'/data_1000_45.npz','./data/'+args.dataset+'/data_1000_46.npz','./data/'+args.dataset+'/data_1000_47.npz','./data/'+args.dataset+'/data_1000_48.npz','./data/'+args.dataset+'/data_1000_49.npz','./data/'+args.dataset+'/data_1000_50.npz','./data/'+args.dataset+'/data_1000_51.npz','./data/'+args.dataset+'/data_1000_52.npz']
                val_data = ['./data/'+args.dataset+'/data_1000_53.npz']
            else:
                pretrain_data = [['./data/'+args.dataset+'/data_1000_0.npz','./data/'+args.dataset+'/data_1000_1.npz','./data/'+args.dataset+'/data_1000_2.npz','./data/'+args.dataset+'/data_1000_3.npz','./data/'+args.dataset+'/data_1000_4.npz','./data/'+args.dataset+'/data_1000_5.npz','./data/'+args.dataset+'/data_1000_6.npz','./data/'+args.dataset+'/data_1000_7.npz','./data/'+args.dataset+'/data_1000_8.npz','./data/'+args.dataset+'/data_1000_9.npz','./data/'+args.dataset+'/data_1000_10.npz','./data/'+args.dataset+'/data_1000_11.npz','./data/'+args.dataset+'/data_1000_12.npz','./data/'+args.dataset+'/data_1000_13.npz','./data/'+args.dataset+'/data_1000_14.npz'],
                                 ['./data/'+args.dataset+'/data_1000_15.npz','./data/'+args.dataset+'/data_1000_16.npz','./data/'+args.dataset+'/data_1000_17.npz','./data/'+args.dataset+'/data_1000_18.npz','./data/'+args.dataset+'/data_1000_19.npz','./data/'+args.dataset+'/data_1000_20.npz','./data/'+args.dataset+'/data_1000_21.npz','./data/'+args.dataset+'/data_1000_22.npz','./data/'+args.dataset+'/data_1000_23.npz','./data/'+args.dataset+'/data_1000_24.npz','./data/'+args.dataset+'/data_1000_25.npz','./data/'+args.dataset+'/data_1000_26.npz','./data/'+args.dataset+'/data_1000_27.npz','./data/'+args.dataset+'/data_1000_28.npz','./data/'+args.dataset+'/data_1000_29.npz'],
                                 ['./data/'+args.dataset+'/data_1000_30.npz','./data/'+args.dataset+'/data_1000_31.npz','./data/'+args.dataset+'/data_1000_32.npz','./data/'+args.dataset+'/data_1000_33.npz','./data/'+args.dataset+'/data_1000_34.npz','./data/'+args.dataset+'/data_1000_35.npz','./data/'+args.dataset+'/data_1000_36.npz','./data/'+args.dataset+'/data_1000_37.npz','./data/'+args.dataset+'/data_1000_38.npz','./data/'+args.dataset+'/data_1000_39.npz','./data/'+args.dataset+'/data_1000_40.npz','./data/'+args.dataset+'/data_1000_41.npz','./data/'+args.dataset+'/data_1000_42.npz','./data/'+args.dataset+'/data_1000_43.npz','./data/'+args.dataset+'/data_1000_44.npz']]
            train_data = ['./data/'+args.dataset+'/data_1000_0.npz','./data/'+args.dataset+'/data_1000_1.npz','./data/'+args.dataset+'/data_1000_2.npz','./data/'+args.dataset+'/data_1000_3.npz','./data/'+args.dataset+'/data_1000_4.npz','./data/'+args.dataset+'/data_1000_5.npz','./data/'+args.dataset+'/data_1000_6.npz','./data/'+args.dataset+'/data_1000_7.npz','./data/'+args.dataset+'/data_1000_8.npz','./data/'+args.dataset+'/data_1000_9.npz','./data/'+args.dataset+'/data_1000_10.npz','./data/'+args.dataset+'/data_1000_11.npz','./data/'+args.dataset+'/data_1000_12.npz','./data/'+args.dataset+'/data_1000_13.npz','./data/'+args.dataset+'/data_1000_14.npz']
            val_data = ['./data/'+args.dataset+'/data_1000_15.npz','./data/'+args.dataset+'/data_1000_16.npz','./data/'+args.dataset+'/data_1000_17.npz','./data/'+args.dataset+'/data_1000_18.npz','./data/'+args.dataset+'/data_1000_19.npz','./data/'+args.dataset+'/data_1000_20.npz','./data/'+args.dataset+'/data_1000_21.npz','./data/'+args.dataset+'/data_1000_22.npz','./data/'+args.dataset+'/data_1000_23.npz','./data/'+args.dataset+'/data_1000_24.npz','./data/'+args.dataset+'/data_1000_25.npz','./data/'+args.dataset+'/data_1000_26.npz','./data/'+args.dataset+'/data_1000_27.npz','./data/'+args.dataset+'/data_1000_28.npz','./data/'+args.dataset+'/data_1000_29.npz']
        
    # Vocab
    if args.new_vocab:
        if args.dataset == 'reference':
            langs = np.array([])
            for files in pretrain_data:
                for file in files:
                    d = data.load_raw_data(file)
                    langs = np.append(langs, d['langs'])
            vocab = data.init_vocab(langs)
            torch.save(vocab,'./models/single/vocab.pt')
        else:
            print('see colors_and_chairs file')
    else:
        vocab = torch.load('./models/'+str(args.dataset)+'/vocab.pt')

    speaker_embs = nn.Embedding(len(vocab['w2i'].keys()), 50)
    listener_embs = nn.Embedding(len(vocab['w2i'].keys()), 50)
    
    # Model
    speaker_vision = vision.Conv4()
    listener_vision = vision.Conv4()
    if args.srr:
        speaker = models.LiteralSpeaker(speaker_vision, speaker_embs)
    else:
        speaker = models.Speaker(speaker_vision, speaker_embs, args.game_type)
    listener = models.Listener(listener_vision, listener_embs)
    if args.cuda:
        speaker = speaker.cuda()
        listener = listener.cuda()
            
    # Optimization
    optimizer = optim.Adam(list(speaker.parameters())+list(listener.parameters()),lr=args.lr)
    loss = nn.CrossEntropyLoss()

    # Metrics
    metrics = init_metrics()
    all_metrics = []

    # Pretrain
    if args.pretrain or args.srr:
        if args.new:
            output_file = './models/'+str(args.dataset)+'/pretrained-listener-1.pt'
            file = ['./data/'+args.dataset+'/data_1000_27.npz','./data/'+args.dataset+'/data_1000_28.npz','./data/'+args.dataset+'/data_1000_29.npz','./data/'+args.dataset+'/data_1000_30.npz','./data/'+args.dataset+'/data_1000_31.npz','./data/'+args.dataset+'/data_1000_32.npz','./data/'+args.dataset+'/data_1000_33.npz','./data/'+args.dataset+'/data_1000_34.npz','./data/'+args.dataset+'/data_1000_35.npz','./data/'+args.dataset+'/data_1000_36.npz','./data/'+args.dataset+'/data_1000_37.npz','./data/'+args.dataset+'/data_1000_38.npz','./data/'+args.dataset+'/data_1000_39.npz','./data/'+args.dataset+'/data_1000_40.npz','./data/'+args.dataset+'/data_1000_41.npz','./data/'+args.dataset+'/data_1000_42.npz','./data/'+args.dataset+'/data_1000_43.npz','./data/'+args.dataset+'/data_1000_44.npz','./data/'+args.dataset+'/data_1000_45.npz','./data/'+args.dataset+'/data_1000_46.npz','./data/'+args.dataset+'/data_1000_47.npz','./data/'+args.dataset+'/data_1000_48.npz','./data/'+args.dataset+'/data_1000_49.npz','./data/'+args.dataset+'/data_1000_50.npz','./data/'+args.dataset+'/data_1000_51.npz','./data/'+args.dataset+'/data_1000_52.npz','./data/'+args.dataset+'/data_1000_53.npz']
            for epoch in range(args.epochs):
                # Train
                data_file = file[0:len(file)-1]
                train_metrics, _ = run(epoch, data_file, 'train', 'pretrain', speaker, listener, optimizer, loss, vocab, args.batch_size, args.cuda)
                # Validation
                data_file = [file[-1]]
                val_metrics, _ = run(epoch, data_file, 'val', 'pretrain', speaker, listener, optimizer, loss, vocab, args.batch_size, args.cuda)
                # Update metrics, prepending the split name
                for metric, value in train_metrics.items():
                    metrics['train_{}'.format(metric)].append(value)
                for metric, value in val_metrics.items():
                    metrics['val_{}'.format(metric)].append(value)
                metrics['current_epoch'] = epoch
                # Use validation accuracy to choose the best model
                is_best = val_metrics['acc'] > metrics['best_acc']
                if is_best:
                    metrics['best_acc'] = val_metrics['acc']
                    metrics['best_loss'] = val_metrics['loss']
                    metrics['best_epoch'] = epoch
                    best_listener = copy.deepcopy(listener)
                print(epoch)
                print(metrics)
            # Save the best model
            listener = best_listener
            torch.save(listener, output_file)
        else:
            listener = torch.load('./models/'+str(args.dataset)+'/pretrained-listener-0.pt')
            listener_val = torch.load('./models/'+str(args.dataset)+'/pretrained-listener-1.pt')
            #speaker = torch.load('./models/'+args.dataset+'/pretrained_len_001_speaker.pt')
            
    # Train
    if args.srr:
        for epoch in range(args.epochs):
            # Train
            train_metrics, _ = run(epoch, train_data, 'train', 'literal', speaker, listener, optimizer, loss, vocab, args.batch_size, args.cuda, game_type = args.game_type, lm_wt = args.lm_wt)
            # Validation
            val_metrics, _ = run(epoch, val_data, 'val', 'literal', speaker, listener_val, optimizer, loss, vocab, args.batch_size, args.cuda, game_type = args.game_type, lm_wt = args.lm_wt)
            # Update metrics, prepending the split name
            for metric, value in train_metrics.items():
                metrics['train_{}'.format(metric)].append(value)
            for metric, value in val_metrics.items():
                metrics['val_{}'.format(metric)].append(value)
            metrics['current_epoch'] = epoch
            # Use validation accuracy to choose the best model
            is_best = val_metrics['acc'] > metrics['best_acc']
            if is_best:
                metrics['best_acc'] = val_metrics['acc']
                metrics['best_loss'] = val_metrics['loss']
                metrics['best_epoch'] = epoch
                best_speaker = copy.deepcopy(speaker)
            print(epoch)
            print(metrics)
            metrics_last = {k: v[-1] if isinstance(v, list) else v
                            for k, v in metrics.items()}
            all_metrics.append(metrics_last)
            pd.DataFrame(all_metrics).to_csv(args.metrics_file, index=False)
        # Save the best model
        speaker = best_speaker
        if args.generalization != None:
            torch.save(speaker, './models/single/'+args.generalization+'_literal_speaker.pt')
        else:
            torch.save(speaker, './models/'+args.dataset+'/literal_speaker.pt')
    else:
        for epoch in range(args.epochs):
            if args.pretrain:
                train_metrics, _ = run(epoch, train_data, 'train', 'pretrained', speaker, listener, optimizer, loss, vocab, args.batch_size, args.cuda, args.game_type, lm_wt = args.lm_wt, activation = args.activation, dataset = args.dataset, penalty = args.penalty, tau = args.tau)
                val_metrics, _ = run(epoch, val_data, 'val', 'pretrained', speaker, listener_val, optimizer, loss, vocab, args.batch_size, args.cuda, args.game_type, lm_wt = args.lm_wt, activation = args.activation, dataset = args.dataset, penalty = args.penalty, tau = args.tau)
            else:
                train_metrics, _ = run(epoch, train_data, 'train', 'cotrained', speaker, listener, optimizer, loss, vocab, args.batch_size, args.cuda, args.game_type, dataset = args.dataset, penalty = args.penalty)
                val_metrics, _ = run(epoch, val_data, 'val', 'cotrained', speaker, listener, optimizer, loss, vocab, args.batch_size, args.cuda, args.game_type, dataset = args.dataset, penalty = args.penalty)
            # Update metrics, prepending the split name
            for metric, value in train_metrics.items():
                metrics['train_{}'.format(metric)].append(value)
            for metric, value in val_metrics.items():
                metrics['val_{}'.format(metric)].append(value)
            metrics['current_epoch'] = epoch
            # Use validation accuracy to choose the best model
            is_best = val_metrics['acc'] > metrics['best_acc']
            if is_best:
                metrics['best_acc'] = val_metrics['acc']
                metrics['best_loss'] = val_metrics['loss']
                metrics['best_epoch'] = epoch
                best_speaker = copy.deepcopy(speaker)
                best_listener = copy.deepcopy(listener)
            print(epoch)
            print(metrics)
            metrics_last = {k: v[-1] if isinstance(v, list) else v
                            for k, v in metrics.items()}
            all_metrics.append(metrics_last)
            pd.DataFrame(all_metrics).to_csv(args.metrics_file, index=False)
        # Save the best model
        if args.pretrain:
            if args.generalization != None:
                torch.save(speaker, './models/'+args.dataset+'/'+args.generalization+'_pretrained_speaker.pt')
            elif args.activation != None:
                torch.save(best_speaker, './models/'+args.dataset+'/pretrained_speaker_'+args.activation+'.pt')
            #elif args.tau != 1:
            #    torch.save(best_speaker, './models/'+args.dataset+'/pretrained_speaker_'+str(int(args.tau*100))+'.pt')
            else:
                torch.save(best_speaker, './models/'+args.dataset+'/pretrained_len_01_speaker.pt')
        else:
            torch.save(best_speaker, './models/single/speaker.pt')
            torch.save(best_listener, './models/single/listener.pt')