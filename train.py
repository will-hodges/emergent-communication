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

import models
import vision
import util
from data import ShapeWorld
from run import run
import data
from shapeworld import SHAPES, COLORS, VOCAB

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
    parser.add_argument('--game_type', choices=['concept', 'reference'], default='reference', type=str)
    parser.add_argument('--srr', action='store_true')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-3, type=int)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--new_vocab', action='store_true')
    args = parser.parse_args()
    
    # Data
    if args.pretrain or args.srr:
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
    train_data = ['./data/single/reference-1000-60.npz','./data/single/reference-1000-61.npz','./data/single/reference-1000-62.npz','./data/single/reference-1000-63.npz','./data/single/reference-1000-64.npz']
    val_data = ['./data/single/reference-1000-65.npz','./data/single/reference-1000-66.npz','./data/single/reference-1000-67.npz','./data/single/reference-1000-68.npz','./data/single/reference-1000-69.npz']
    
    # Vocab
    speaker_embs = nn.Embedding(4+len(VOCAB), 50)
    listener_embs = nn.Embedding(4+len(VOCAB), 50)
    if args.new_vocab:
        langs = np.array([])
        for files in pretrain_data:
            for file in files:
                d = data.load_raw_data(file)
                langs = np.append(langs, d['langs'])
        vocab = data.init_vocab(langs)
        torch.save(vocab,'vocab.pt')
    else:
        vocab = torch.load('vocab.pt')

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

    # Pretrain
    if args.pretrain:
        if args.new:
            file = ['./data/single/reference-1000-70.npz','./data/single/reference-1000-71.npz','./data/single/reference-1000-72.npz','./data/single/reference-1000-73.npz','./data/single/reference-1000-74.npz']
            output_file = 'pretrained-listener-01.pt'
            for epoch in range(args.epochs):
                # Train
                data_file = file[0:len(file)-1]
                train_metrics, _ = run(epoch, data_file, 'train', 'pretrain', speaker, listener, optimizer, loss, vocab, args.batch_size, args.cuda, args.cuda)
                # Validation
                data_file = [file[-1]]
                val_metrics, _ = run(epoch, data_file, 'val', 'pretrain', speaker, listener, optimizer, loss, vocab, args.batch_size, args.cuda, args.cuda)
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
            """"
            output_files = ['pretrained-listener-0.pt','pretrained-listener-1.pt','pretrained-listener-2.pt','pretrained-listener-3.pt','pretrained-listener-4.pt','pretrained-listener-5.pt','pretrained-listener-6.pt','pretrained-listener-7.pt','pretrained-listener-8.pt','pretrained-listener-9.pt','pretrained-listener-10.pt','pretrained-listener-01.pt']
            for file, output_file in zip(pretrain_data, output_files):
                for epoch in range(args.epochs):
                    # Train
                    data_file = file[0:len(file)-1]
                    train_metrics, _ = run(epoch, data_file, 'train', 'pretrain', speaker, listener, optimizer, loss, vocab, args.batch_size, args.cuda, args.cuda)
                    # Validation
                    data_file = [file[-1]]
                    val_metrics, _ = run(epoch, data_file, 'val', 'pretrain', speaker, listener, optimizer, loss, vocab, args.batch_size, args.cuda, args.cuda)
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
                torch.save(listener, output_file)"""
        else:
            listener = torch.load('pretrained-listener-0.pt')
            listener_val = torch.load('pretrained-listener-1.pt')
            
    # Train
    if args.srr:
        for epoch in range(args.epochs):
            # Train
            train_metrics, _ = run(epoch, train_data, 'train', 'literal', speaker, listener, optimizer, loss, vocab, args.batch_size, args.cuda, game_type = args.game_type)
            # Validation
            val_metrics, _ = run(epoch, val_data, 'val', 'literal', speaker, listener_val, optimizer, loss, vocab, args.batch_size, args.cuda, game_type = args.game_type)
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
        # Save the best model
        speaker = best_speaker
        torch.save(speaker, 'literal_speaker.pt')
    else:
        for epoch in range(args.epochs):
            # Train
            train_metrics, _ = run(epoch, train_data, 'train', 'pragmatic', speaker, listener, optimizer, loss, vocab, args.batch_size, args.cuda, args.game_type)
            # Validation
            if args.pretrain:
                val_metrics, _ = run(epoch, val_data, 'val', 'pragmatic', speaker, listener_val, optimizer, loss, vocab, args.batch_size, args.cuda, args.game_type)
            else:
                val_metrics, _ = run(epoch, val_data, 'val', 'pragmatic', speaker, listener, optimizer, loss, vocab, args.batch_size, args.cuda, args.game_type)
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
        # Save the best model
        if args.pretrain:
            torch.save(best_speaker, 'pretrained_speaker.pt')
        else:
            torch.save(best_speaker, 'speaker.pt')
            torch.save(best_listener, 'listener.pt')