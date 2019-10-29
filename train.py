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
import data
from shapeworld import SHAPES, COLORS

# Logging
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

VOCAB = ['gray', 'shape', 'blue', 'square', 'circle', 'green', 'red', 'rectangle', 'yellow', 'ellipse', 'white']

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

def sample_ref_game(img, y, games_per_batch = 10):
    sampled_img = []
    sampled_y = []
    for batch in range(len(img)):
        game_img = []
        game_y = []
        for game in range(games_per_batch):
            y_batch = y[batch]
            img_batch = img[batch]

            pos_idx = np.nonzero(y_batch)
            neg_idx = np.nonzero(y_batch-1)
            pos_idx = pos_idx[random.sample(range(len(pos_idx)), 1)]
            neg_idx = neg_idx[random.sample(range(len(neg_idx)), 2)]

            game_img.append(torch.cat([img_batch[pos_idx[0]], img_batch[neg_idx[0]], img_batch[neg_idx[1]]],dim=0).unsqueeze(0))
            game_y.append(torch.cat([y_batch[pos_idx[0]], y_batch[neg_idx[0]], y_batch[neg_idx[1]]],dim=0).unsqueeze(0))

        sampled_img.append(torch.cat(game_img, dim=0).unsqueeze(0))
        sampled_y.append(torch.cat(game_y, dim=0).unsqueeze(0))

    sampled_img = torch.cat(sampled_img, dim=0)
    sampled_y = torch.cat(sampled_y, dim=0)
    return sampled_img, sampled_y
    
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

def run(data_file, split, run_type, speaker, listener, optimizer, loss, game_type = 'reference'):
    """
    Run the model for a single epoch.

    Parameters
    ----------
    split : ``str``
        The dataloader split to use. Also determines model behavior if e.g.
        ``split == 'train'`` then model will be in train mode/optimizer will be
        run.
    epoch : ``int``
        current epoch
    model : ``torch.nn.Module``
        the model you are training/evaling
    optimizer : ``torch.nn.optim.Optimizer``
        the optimizer
    loss : ``torch.nn.loss``
        the loss function
    dataloaders : ``dict[str, torch.utils.data.DataLoader]``
        Dictionary of dataloaders whose keys are the names of the ``split``s
        and whose values are the corresponding dataloaders
    args : ``argparse.Namespace``
        Arguments for this experiment run
    random_state : ``np.random.RandomState``
        The numpy random state in case anything stochastic happens during the
        run

    Returns
    -------
    metrics : ``dict[str, float]``
        Metrics from this run; keys are statistics and values are their average
        values across the batches
    """
    outputs = []
    if split == 'train':
        if run_type == 'literal':
            speaker.train()
        if run_type == 'pretrain':
            listener.train()
        else:
            speaker.train()
            listener.train()
        context = contextlib.suppress()
    else:
        if run_type != 'pretrain':
            speaker.eval()
        if run_type != 'literal':
            listener.eval()
        context = torch.no_grad()  # Do not evaluate gradients for efficiency

    # Initialize your average meters to keep track of the epoch's running average
    measures = ['loss', 'acc']
    meters = {m: util.AverageMeter() for m in measures}

    with context:
        for file in data_file:
            d = data.load_raw_data(file)
            dataloader = DataLoader(ShapeWorld(d, vocab), batch_size=args.batch_size, shuffle=True)
            for batch_i, (img, y, lang) in enumerate(dataloader):
                if run_type == 'literal' or run_type == 'pretrain' or game_type == 'reference' or run_type == 'sample':
                    y = y.argmax(1) # convert from onehot
                batch_size = img.shape[0]

                # Convert to float
                img = img.float()

                # Refresh the optimizer
                if split == 'train':
                    optimizer.zero_grad()

                if run_type == 'literal' or run_type == 'pretrain':
                    max_len = 20
                    length = torch.tensor([np.count_nonzero(t) for t in lang.cpu()], dtype=np.int)
                    lang = F.one_hot(lang, num_classes = 4+len(VOCAB))
                    lang = F.pad(lang,(0,0,0,max_len-lang.shape[1])).float()
                    for B in range(lang.shape[0]):
                        for L in range(lang.shape[1]):
                            if lang[B][L].sum() == 0:
                                lang[B][L][0] = 1

                if args.cuda:
                    img = img.cuda()
                    y = y.cuda()
                    if run_type == 'literal' or run_type == 'pretrain':
                        lang.cuda()
                        length = length.cuda()
                
                # Forward pass
                if run_type == 'pretrain':
                    lis_scores = listener(img, lang, length)
                elif run_type == 'literal':
                    out = lang.reshape(lang.shape[0]*lang.shape[1],-1)
                    outputs = out.argmax(1)
                    outputs = outputs.reshape(lang.shape[0],lang.shape[1])
                    hypo_out = speaker(img, lang, length, y)
                elif run_type == 'sample':
                    lang, lang_length = speaker.sample(img, y, greedy = True)
                else:
                    lang, lang_length = speaker(img, y)

                # Evaluate loss and accuracy
                if run_type == 'pretrain':
                    this_loss = loss(lis_scores, y)
                    lis_pred = lis_scores.argmax(1)
                    this_acc = (lis_pred == y).float().mean().item()

                    if split == 'train':
                        # SGD step
                        this_loss.backward()
                        optimizer.step()
                    
                    meters['loss'].update(this_loss, batch_size)
                    meters['acc'].update(this_acc, batch_size)
                elif run_type == 'literal':
                    hint_seq = lang.cuda()
                    hypo_out = hypo_out[:, :-1].contiguous()
                    hint_seq = hint_seq[:, 1:].contiguous()
                    
                    seq_len = hypo_out.size(1)
                    
                    hypo_out_2d = hypo_out.view(batch_size * seq_len, 4+len(VOCAB))
                    hint_seq_2d = hint_seq.long().view(batch_size * seq_len, 4+len(VOCAB))
                    hypo_loss = loss(hypo_out_2d.cuda(), torch.max(hint_seq_2d, 1)[1])

                    if split == 'train':
                        # SGD step
                        hypo_loss.backward()
                        optimizer.step()
                        
                    hypo_acc = (hypo_out_2d.argmax(1)==hint_seq_2d.argmax(1)).float().mean().item()

                    meters['loss'].update(hypo_loss, batch_size)
                    meters['acc'].update(hypo_acc, batch_size)
                    outputs = [hypo_out_2d.argmax(1),hint_seq_2d.argmax(1)]
                else:
                    if game_type == 'reference':
                        lis_scores = listener(img, lang, lang_length)

                        # Evaluate loss and accuracy
                        this_loss = loss(lis_scores, y.long())
                        lis_pred = lis_scores.argmax(1)
                        correct = (lis_pred == y)
                        this_acc = correct.float().mean().item()

                        if split == 'train':
                            # SGD step
                            this_loss.backward()
                            optimizer.step()

                        meters['loss'].update(this_loss, batch_size)
                        meters['acc'].update(this_acc, batch_size)
                        
                        out = lang.reshape(lang.shape[0]*lang.shape[1],-1)
                        outputs = out.argmax(1)
                        outputs = outputs.reshape(lang.shape[0],lang.shape[1])

                    if game_type == 'concept':
                        games_per_batch = 5
                        sampled_img, sampled_y = sample_ref_game(img, y, games_per_batch = games_per_batch)
                        for game in range(games_per_batch):
                            lis_scores = listener(torch.squeeze(sampled_img[:,game]), lang, lang_length)

                            # Evaluate loss and accuracy
                            y = torch.squeeze(sampled_y[:,game]).argmax(1)
                            this_loss = loss(lis_scores, y)
                            lis_pred = lis_scores.argmax(1)
                            correct = (lis_pred == y)
                            this_acc = correct.float().mean().item()

                            if split == 'train':
                                # SGD step
                                this_loss.backward(retain_graph=True)
                                optimizer.step()

                            meters['loss'].update(this_loss, batch_size)
                            meters['acc'].update(this_acc, batch_size)   
    
    metrics = compute_average_metrics(meters)
    return metrics, outputs


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(description='Train', formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--new', action='store_true')
    parser.add_argument('--game_type', choices=['concept', 'reference'], default='reference', type=str)
    parser.add_argument('--data_type', choices=['single', 'spatial'], default='single', type=str)
    parser.add_argument('--rsa', action='store_true')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--pretrain_data_file', default=None, type=str)
    parser.add_argument('--train_data_file', default=None, type=str)
    parser.add_argument('--val_data_file', default=None, type=str)
    parser.add_argument('--test_data_file', default=None, type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-3, type=int)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--new_vocab', action='store_true')
    args = parser.parse_args()
    
    # Data
    if (args.pretrain or args.rsa) and args.pretrain_data_file == None:
        if args.data_type == 'single':
            args.pretrain_data_file = ['../data/single/reference-1000-6.npz','../data/single/reference-1000-7.npz','../data/single/reference-1000-8.npz','../data/single/reference-1000-9.npz','../data/single/reference-1000-10.npz']
        if args.data_type == 'spatial':
            args.pretrain_data_file = ['../data/spatial/reference-1000-6.npz','../data/spatial/reference-1000-7.npz','../data/spatial/reference-1000-8.npz','../data/spatial/reference-1000-9.npz','../data/spatial/reference-1000-10.npz']
    if args.train_data_file == None:
        if args.game_type == 'reference':
            if args.data_type == 'single':
                args.train_data_file = ['../data/single/reference-1000-1.npz','../data/single/reference-1000-2.npz','../data/single/reference-1000-3.npz','../data/single/reference-1000-4.npz']
            if args.data_type == 'spatial':
                args.train_data_file = ['../data/spatial/reference-1000-1.npz','../data/spatial/reference-1000-2.npz','../data/spatial/reference-1000-3.npz','../data/spatial/reference-1000-4.npz']
        if args.game_type == 'concept':
            if args.data_type == 'single':
                args.train_data_file = ['../data/single/concept-100-1.npz','../data/single/concept-100-2.npz','../data/single/concept-100-3.npz','../data/single/concept-100-4.npz','../data/single/concept-100-5.npz','../data/single/concept-100-6.npz','../data/single/concept-100-7.npz','../data/single/concept-100-8.npz','../data/single/concept-100-9.npz','../data/single/concept-100-10.npz','../data/single/concept-100-11.npz','../data/single/concept-100-12.npz','../data/single/concept-100-13.npz','../data/single/concept-100-14.npz','../data/single/concept-100-15.npz','../data/single/concept-100-16.npz','../data/single/concept-100-17.npz','../data/single/concept-100-18.npz','../data/single/concept-100-19.npz','../data/single/concept-100-20.npz','../data/single/concept-100-21.npz','../data/single/concept-100-22.npz','../data/single/concept-100-23.npz','../data/single/concept-100-24.npz','../data/single/concept-100-25.npz','../data/single/concept-100-26.npz','../data/single/concept-100-27.npz','../data/single/concept-100-28.npz','../data/single/concept-100-29.npz','../data/single/concept-100-30.npz','../data/single/concept-100-31.npz','../data/single/concept-100-32.npz','../data/single/concept-100-33.npz','../data/single/concept-100-34.npz','../data/single/concept-100-35.npz','../data/single/concept-100-36.npz','../data/single/concept-100-37.npz','../data/single/concept-100-38.npz','../data/single/concept-100-39.npz','../data/single/concept-100-40.npz','../data/single/concept-100-41.npz','../data/single/concept-100-42.npz','../data/single/concept-100-43.npz','../data/single/concept-100-44.npz','../data/single/concept-100-45.npz','../data/single/concept-100-46.npz','../data/single/concept-100-47.npz','../data/single/concept-100-48.npz','../data/single/concept-100-49.npz']
            if args.data_type == 'spatial':
                args.train_data_file = ['../data/spatial/concept-1000-1.npz','../data/spatial/concept-1000-2.npz','../data/spatial/concept-1000-3.npz','../data/spatial/concept-1000-4.npz']
    if args.val_data_file == None:
        if args.game_type == 'reference':
            if args.data_type == 'single':
                args.val_data_file = ['../data/single/reference-1000-5.npz']
            if args.data_type == 'spatial':
                args.val_data_file = ['../data/spatial/reference-1000-5.npz']
        if args.game_type == 'concept':
            if args.data_type == 'single':
                args.val_data_file = ['../data/single/concept-100-50.npz']
            if args.data_type == 'spatial':
                args.val_data_file = ['../data/spatial/concept-1000-5.npz']
    if args.test_data_file == None:
        if args.game_type == 'reference':
            if args.data_type == 'single':
                args.test_data_file = ['../data/single/reference-1000-6.npz']
            if args.data_type == 'spatial':
                args.test_data_file = ['../data/spatial/reference-1000-6.npz']
        if args.game_type == 'concept':
            if args.data_type == 'single':
                args.test_data_file = ['../data/single/concept-100-51.npz']
            if args.data_type == 'spatial':
                args.test_data_file = ['../data/spatial/concept-1000-6.npz']
    
    # Vocab
    speaker_embs = nn.Embedding(4 + len(VOCAB), 50)
    listener_embs = nn.Embedding(4 + len(VOCAB), 50)
    if args.new_vocab:
        langs = np.array([])
        for file in args.pretrain_data_file:
            d = data.load_raw_data(file)
            langs = np.append(langs, d['langs'])
        for file in args.train_data_file:
            d = data.load_raw_data(file)
            langs = np.append(langs, d['langs'])
        for file in args.val_data_file:
            d = data.load_raw_data(file)
            langs = np.append(langs, d['langs'])
        for file in args.test_data_file:
            d = data.load_raw_data(file)
            langs = np.append(langs, d['langs'])
        vocab = data.init_vocab(langs)
        if single:
            torch.save(vocab,'single_vocab.pt')
        if spatial:
            torch.save(vocab,'single_vocab.pt')
    else:
        vocab = torch.load('single_vocab.pt')

    # Model
    if args.rsa:
        speaker_vision = vision.Conv4()
        speaker = models.LiteralSpeaker(speaker_vision, speaker_embs)
        listener_vision = vision.Conv4()
        listener = models.Listener(listener_vision, listener_embs)
    else:
        speaker_vision = vision.Conv4()
        speaker = models.Speaker(speaker_vision, speaker_embs, args.game_type)
        listener_vision = vision.Conv4()
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
    if args.pretrain or args.rsa:
        if args.new:
            for epoch in range(args.epochs):
                # Train
                data_file = args.pretrain_data_file[0:len(args.pretrain_data_file)-1]
                train_metrics, outputs = run(data_file, 'train', 'pretrain', speaker, listener, optimizer, loss)
                # Validation
                data_file = [args.pretrain_data_file[-1]]
                val_metrics, outputs = run(data_file, 'val', 'pretrain', speaker, listener, optimizer, loss)

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
                print(val_metrics)
            
            # Save the best model
            listener = best_listener
            torch.save(listener, 'pretrained_listener.pt')
        else:
            listener = torch.load('pretrained_listener.pt')

    # Train
    if args.rsa:
        if args.new:
            for epoch in range(args.epochs):
                # Train
                train_metrics, outputs = run(args.train_data_file, 'train', 'literal', speaker, listener, optimizer, loss, game_type = args.game_type)
                # Validation
                val_metrics, outputs = run(args.val_data_file, 'val', 'literal', speaker, listener, optimizer, loss, game_type = args.game_type)

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
                print(val_metrics)
                
            # Save the best model
            speaker = best_speaker
            torch.save(speaker, 'literal_speaker.pt')
        else:
            speaker = torch.load('literal_speaker.pt')
        
        # Sample and rerank
        test_metrics, outputs = run(args.test_data_file, 'val', 'sample', speaker, listener, optimizer, loss, game_type = args.game_type)
        print(test_metrics)
        print(vocab)
        print(outputs)
    else:
        for epoch in range(args.epochs):
            print(epoch)
            # Train
            train_metrics, outputs = run(args.train_data_file, 'train', 'pragmatic', speaker, listener, optimizer, loss, args.game_type)
            # Validation
            train_metrics, outputs = run(args.val_data_file, 'val', 'pragmatic', speaker, listener, optimizer, loss, args.game_type)

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
            print(val_metrics)

        # Save the best model
        torch.save(best_speaker, 'speaker.pt')
        torch.save(best_listener, 'listener.pt')
