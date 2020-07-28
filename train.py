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
import torchvision
import pandas as pd

import vision
import util
from data import ShapeWorld
import data

from colors import ColorsInContext

from run import run
import models

from glob import glob
import os

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
    
    parser.add_argument('--dataset', default='shapeglot', help='(shapeworld, colors, shapeglot, or chairs)')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--vocab', action='store_true', help='Generate new vocab file')
    parser.add_argument('--s0', action='store_true', help='Train contextual speaker')
    parser.add_argument('--l0', action='store_true', help='Train literal listener')
    parser.add_argument('--sl0', action='store_true', help='Train literal speaker')
    parser.add_argument('--amortized', action='store_true', help='Train amortized speaker')
    parser.add_argument('--activation', default=None)
    parser.add_argument('--penalty', default=None, help='Cost function (length)')
    parser.add_argument('--lmbd', default=0.01, help='Cost function parameter')
    parser.add_argument('--tau', default=1, type=float, help='Softmax temperature')
    parser.add_argument('--save', default='metrics.csv', help='Where to save metrics')
    parser.add_argument('--debug', action='store_true', help='Print metrics on every epoch')
    parser.add_argument('--save_imgs', action='store_true', help='Save one reference game per batch')
    parser.add_argument('--generalization', default=None)
    parser.add_argument('--embed_size', default=300, type=int, help='Size of embedding layers')
    args = parser.parse_args()
    
    # Data
    if args.dataset == 'shapeworld':
        if args.generalization == None:
            data_dir = './data/single/reference-1000-'
            pretrain_data = np.reshape(np.array([data_dir + str(e) + '.npz' for e in range(0,65)]), (3,20)).tolist()
        else:
            data_dir = './data/single/generalization_'+args.generalization+'/reference-1000-'
            pretrain_data = [[data_dir + str(e) + '.npz' for e in range(0,10)]]
        train_data = [data_dir + str(e) + '.npz' for e in range(60,65)]
        val_data = [data_dir + str(e) + '.npz' for e in range(65,70)]
        
    elif args.dataset == 'chairs':
        data_dir = './data/chairs/data_1000_'
        pretrain_data = np.reshape(np.array([data_dir + str(e) + '.npz' for e in range(15,45)]), (3,10)).tolist()
        train_data = [data_dir + str(e) + '.npz' for e in range(0,15)]
        val_data = [data_dir + str(e) + '.npz' for e in range(15,30)]
        
    elif args.dataset == 'shapeglot':
        data_dir = './data/shapeglot/data_1000_'
        pt = np.array(glob(os.path.join(f'data/shapeglot/*_train_*.npz')))[:-2]
        pretrain_data = np.reshape(pt[:30], (3,int(len(pt[:30])/3))).tolist()
        train_data = glob(os.path.join(f'data/shapeglot/*_test_*.npz'))
        val_data = glob(os.path.join(f'data/shapeglot/*_val_*.npz'))
        
    elif args.dataset == 'colors':
        data_dir = './data/colors/data_1000_'
        pretrain_data = np.reshape(np.array([data_dir + str(e) + '.npz' for e in range(15,45)]), (3,10)).tolist()
        train_data = [data_dir + str(e) + '.npz' for e in range(0,15)]
        val_data = [data_dir + str(e) + '.npz' for e in range(15,30)]
    else:
        raise Exception('Dataset '+args.dataset+' is not defined.')
        
    # Load or Generate Vocab
    if args.vocab:
        langs = np.array([])
        for files in pretrain_data:
            for file in files:
                d = data.load_raw_data(file)
                langs = np.append(langs, d['langs'])
        vocab = data.init_vocab(langs)
        torch.save(vocab,'./models/'+args.dataset+'/vocab.pt')
    else:
        vocab = torch.load('./models/'+args.dataset+'/vocab.pt')
    
    # Initialize Speaker and Listener Model
    speaker_embs = nn.Embedding(len(vocab['w2i'].keys()), args.embed_size)
    speaker_vision = vision.Conv4()
    if args.s0: # s0 is actually a contextual speaker, *not* a literal speaker. that's sl0
        speaker = models.LiteralSpeaker(speaker_vision, speaker_embs, contextual=True)
    elif args.sl0:
        speaker = models.LiteralSpeaker(speaker_vision, speaker_embs, contextual=False)
    else:
        speaker = models.Speaker(speaker_vision, speaker_embs)
    listener_embs = nn.Embedding(len(vocab['w2i'].keys()), args.embed_size)
    listener_vision = vision.ResNet18()
    listener = models.Listener(listener_vision, listener_embs)
    if args.cuda:
        speaker = speaker.cuda()
        listener = listener.cuda()
        
    # Optimization
    optimizer = optim.Adam(list(speaker.parameters())+list(listener.parameters()),lr=args.lr)
    loss = nn.CrossEntropyLoss()

    # Initialize Metrics
    metrics = init_metrics()
    all_metrics = []
    last_five = []

    # Pretrain Literal Listener
    if args.l0:
        if args.generalization:
            output_dir = './models/' + args.dataset + '/'+args.generalization+'_pretrained_listener_'
            output_files = [output_dir+'0.pt', output_dir+'1.pt']
        else:
            output_dir = './models/'+args.dataset+'/pretrained_listener_'
            output_files = [output_dir+'0.pt', output_dir+'1.pt', output_dir+'2.pt']
        

        for file, output_file in zip(pretrain_data,output_files):
            # Reinitialize metrics, listener model, and optimizer
            metrics = init_metrics()
            all_metrics = []
            listener_embs = nn.Embedding(len(vocab['w2i'].keys()), args.embed_size)
            listener_vision = vision.ResNet18()
            listener = models.Listener(listener_vision, listener_embs)
            if args.cuda:
                listener = listener.cuda()
            optimizer = optim.Adam(list(listener.parameters()),lr=args.lr)
        
            for epoch in range(args.epochs):
                # Train one epoch
                if args.dataset != 'shapeglot':
                    data_file = file[0:len(file)-1]
                else:
                    data_file = file
                    
                print('beginning training!')
                
                train_metrics, _ = run(data_file, 'train', 'l0', None, listener, optimizer, loss, vocab, args.batch_size, args.cuda, debug = args.debug, save_imgs = args.save_imgs, dataset=args.dataset)
                
                print("train done; val beginning")

                # Validate
                data_file = [file[-1]]
                val_metrics, _ = run(data_file, 'val', 'l0', None, listener, optimizer, loss, vocab, args.batch_size, args.cuda, debug = args.debug, save_imgs = args.save_imgs, dataset=args.dataset)
                
                print('val done')

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
                    
                all_metrics.append(metrics)
                
                if args.debug:
                    print(metrics)
                
                '''# Early stopping
                if (len(last_five) == 5):
                    last_five.pop(0)
                    last_five.append(round(val_metrics['acc'], 2))
                else:
                    last_five.append(round(val_metrics['acc'], 2))
                    
                if round(val_metrics['acc'], 2) == 1.0:
                    break
                
                if len(last_five) == 5:
                    if (last_five[0] - last_five[4]) <= 0:
                        # If not decreasing for last five
                        break'''

            # Save the best model
            literal_listener = best_listener
            torch.save(literal_listener, output_file)
            
            metrics_last = {k: v[-1] if isinstance(v, list) else v
                            for k, v in metrics.items()}
            all_metrics.append(metrics_last)
            pd.DataFrame(all_metrics).to_csv(args.save, index=False)
            
    # Load Literal Listener
    if args.amortized or args.s0 or args.sl0:
        if args.generalization:
            literal_listener = torch.load('./models/' + args.dataset + '/'+args.generalization+'_pretrained_listener_0.pt')
            literal_listener_val = torch.load('./models/' + args.dataset + '/'+args.generalization+'_pretrained_listener_1.pt')
        else:
            literal_listener = torch.load('./models/'+args.dataset+'/pretrained_listener_0.pt')
            literal_listener_val = torch.load('./models/'+args.dataset+'/pretrained_listener_1.pt')
            
    # Train Literal Speaker
    if args.sl0:
        for epoch in range(args.epochs):
            # Train one epoch
            train_metrics, _ = run(train_data, 'train', 'sl0', speaker, literal_listener, optimizer, loss, vocab, args.batch_size, args.cuda, lmbd = args.lmbd, debug = args.debug, save_imgs = args.save_imgs, dataset=args.dataset)
            
            # Validate
            val_metrics, _ = run(val_data, 'val', 'sl0', speaker, literal_listener_val, optimizer, loss, vocab, args.batch_size, args.cuda, lmbd = args.lmbd, debug = args.debug, save_imgs = args.save_imgs, dataset=args.dataset)
            
            # Update metrics, prepending the split name
            for metric, value in train_metrics.items():
                metrics['train_{}'.format(metric)].append(value)
            for metric, value in val_metrics.items():
                metrics['val_{}'.format(metric)].append(value)
            metrics['current_epoch'] = epoch

            # Use validation accuracy to choose the best model
            # THIS SHOULD BE val_metrics! Setting to train_metrics 
            # Overfits to the training data
            is_best = train_metrics['acc'] > metrics['best_acc']
            if is_best:
                metrics['best_acc'] = train_metrics['acc']
                metrics['best_loss'] = train_metrics['loss']
                metrics['best_epoch'] = epoch
                best_speaker = copy.deepcopy(speaker)

            if args.debug:
                print(metrics)
                
            ''' # Early stopping
            if (len(last_five) == 5):
                last_five.pop(0)
                last_five.append(round(val_metrics['acc'], 2))
            else:
                last_five.append(round(val_metrics['acc'], 2))
            
            if round(val_metrics['acc'], 2) == 1.0:
                break
            
            if len(last_five) == 5:
                if (last_five == sorted(last_five, reverse=True)):
                    # If not decreasing for last five
                    break'''

            # Store metrics
            metrics_last = {k: v[-1] if isinstance(v, list) else v
                            for k, v in metrics.items()}
            all_metrics.append(metrics_last)
            pd.DataFrame(all_metrics).to_csv(args.save, index=False)
        # Save the best model
        if args.generalization:
            torch.save(best_speaker, './models/' + args.datset + '/'+args.generalization+'_actual_literal_speaker.pt')
        else:
            torch.save(best_speaker, './models/'+args.dataset+'/actual_literal_speaker.pt')

    # Train Contextual Speaker
    if args.s0:
        for epoch in range(args.epochs):
            # Train one epoch
            train_metrics, _ = run(train_data, 'train', 's0', speaker, literal_listener, optimizer, loss, vocab, args.batch_size, args.cuda, lmbd = args.lmbd, debug = args.debug, save_imgs = args.save_imgs)
            
            # Validate
            val_metrics, _ = run(val_data, 'val', 's0', speaker, literal_listener_val, optimizer, loss, vocab, args.batch_size, args.cuda, lmbd = args.lmbd, debug = args.debug, save_imgs = args.save_imgs)
            
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

            if args.debug:
                print(metrics)
                
            '''# Early stopping
            if (len(last_five) == 5):
                last_five.pop(0)
                last_five.append(round(val_metrics['acc'], 2))
            else:
                last_five.append(round(val_metrics['acc'], 2))
            
            if round(val_metrics['acc'], 2) == 1.0:
                break
            
            if len(last_five) == 5:
                if (last_five == sorted(last_five, reverse=True)):
                    # If not decreasing for last five
                    break'''

            # Store metrics
            metrics_last = {k: v[-1] if isinstance(v, list) else v
                            for k, v in metrics.items()}
            all_metrics.append(metrics_last)
            pd.DataFrame(all_metrics).to_csv(args.save, index=False)

        # Save the best model
        if args.generalization:
            torch.save(best_speaker, './models/' + args.datset + '/'+args.generalization+'_literal_speaker.pt')
        else:
            torch.save(best_speaker, './models/'+args.dataset+'/literal_speaker.pt')
    
    # Train Amortized Speaker
    if args.amortized:
        for epoch in range(args.epochs):
            # Train one epoch
            train_metrics, _ = run(train_data, 'train', 'amortized', speaker, literal_listener, optimizer, loss, vocab, args.batch_size, args.cuda, lmbd = args.lmbd, activation = args.activation, dataset = args.dataset, penalty = args.penalty, tau = args.tau, debug = args.debug, save_imgs = args.save_imgs)
            
            # Validate
            val_metrics, _ = run(val_data, 'val', 'amortized', speaker, literal_listener_val, optimizer, loss, vocab, args.batch_size, args.cuda, lmbd = args.lmbd, activation = args.activation, dataset = args.dataset, penalty = args.penalty, tau = args.tau, debug = args.debug, save_imgs = args.save_imgs)

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

            if args.debug:
                print(metrics)
                
                
            '''# Early stopping
            if (len(last_five) == 5):
                last_five.pop(0)
                last_five.append(round(val_metrics['acc'], 2))
            else:
                last_five.append(round(val_metrics['acc'], 2))
            
            if round(val_metrics['acc'], 2) == 1.0:
                break
            
            if len(last_five) == 5:
                if (last_five == sorted(last_five, reverse=True)):
                    # If not decreasing for last five
                    break'''

            # Store metrics
            metrics_last = {k: v[-1] if isinstance(v, list) else v
                            for k, v in metrics.items()}
            all_metrics.append(metrics_last)
            pd.DataFrame(all_metrics).to_csv(args.save, index=False)

        # Save the best model
        try:
            if args.generalization:
                if args.activation == 'multinomial':
                    torch.save(best_speaker, './models/' + args.dataset + '/'+args.generalization+'_pretrained_speaker_multinomial.pt')
                else:
                    torch.save(best_speaker, './models/' + args.dataset + '/'+args.generalization+'_pretrained_speaker.pt')
            else:
                if args.activation == 'multinomial':
                    torch.save(best_speaker, './models/'+args.dataset+'/pretrained_speaker_multinomial.pt')
                elif args.penalty == 'length':
                    torch.save(best_speaker, './models/'+args.dataset+'/pretrained_speaker_penalty_'+str(args.lmbd).replace('.','')+'_2.pt')
                else:
                    torch.save(best_speaker, './models/'+args.dataset+'/pretrained_speaker.pt')
        except:
            random_file = str(np.random.randint(0,1000))
            print('failed saving, now saving at '+random_file+'.pt')
            torch.save(best_speaker, './models/'+random_file+'.pt')


