"""
Train an RNN decoder to make binary predictions;
then train an RNN language model to generate sequences
"""


import contextlib
import random
from collections import defaultdict
import copy

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

def pretrain(split,
        epoch,
        listener,
        optimizer,
        loss,
        dataloaders,
        random_state=None,
        max_len = 20):
    """
    Run pretraining on the listener model for a single epoch.

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
    training = split == 'train'
    dataloader = dataloaders[split]
    if training:
        listener.train()
        context = contextlib.suppress()
    else:
        listener.eval()
        context = torch.no_grad()  # Do not evaluate gradients for efficiency

    # Initialize your average meters to keep track of the epoch's running average
    measures = ['loss', 'acc']
    meters = {m: util.AverageMeter() for m in measures}

    with context:
        for batch_i, (img, y_onehot, lang) in enumerate(dataloader):
            y = y_onehot.argmax(1)
            batch_size = img.shape[0]
            
            # Convert to float
            img = img.float()

            # Refresh the optimizer
            if training:
                optimizer.zero_grad()
            
            lang_length = torch.tensor([np.count_nonzero(t) for t in lang], dtype=np.int)
            lang = F.one_hot(lang, num_classes=4 + len(VOCAB))
            lang = F.pad(lang,(0,0,0,max_len-lang.shape[1])).float()
            
            if args.cuda:
                img = img.cuda()
                y = y.cuda()
                lang = lang.cuda()
                lang_length = lang_length.cuda()

            lis_scores = listener(img, lang, lang_length)

            # Evaluate loss and accuracy
            this_loss = loss(lis_scores, y)
            lis_pred = lis_scores.argmax(1)
            this_acc = (lis_pred == y).float().mean().item()

            if training:
                # SGD step
                this_loss.backward()
                optimizer.step()
            
            meters['loss'].update(this_loss, batch_size)
            meters['acc'].update(this_acc, batch_size)

    metrics = compute_average_metrics(meters)
    return metrics

def run(split,
        epoch,
        speaker,
        listener,
        optimizer,
        loss,
        dataloaders,
        pretrain,
        game,
        random_state=None):
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
    training = split == 'train'
    dataloader = dataloaders[split]
    if training:
        speaker.train()
        if pretrain:
            for param in listener.parameters():
                param.requires_grad = False
        listener.train()
        context = contextlib.suppress()
    else:
        speaker.eval()
        listener.eval()
        context = torch.no_grad()  # Do not evaluate gradients for efficiency

    # Initialize your average meters to keep track of the epoch's running average
    measures = ['loss', 'acc']
    meters = {m: util.AverageMeter() for m in measures}

    with context:
        for batch_i, (img, y, lang) in enumerate(dataloader):
            if game == 'reference':
                y = y.argmax(1) # convert from onehot
            batch_size = img.shape[0]

            # Convert to float
            img = img.float()
            if args.cuda:
                img = img.cuda()
                y = y.cuda()

            # Refresh the optimizer
            if training:
                optimizer.zero_grad()

            # Forward pass
            lang, lang_length = speaker(img, y)

            if game == 'reference':
                lis_scores = listener(img, lang, lang_length)
                
                # Evaluate loss and accuracy
                this_loss = loss(lis_scores, y.long())
                lis_pred = lis_scores.argmax(1)
                this_acc = (lis_pred == y).float().mean().item()
                
                if training:
                    # SGD step
                    this_loss.backward()
                    optimizer.step()

                meters['loss'].update(this_loss, batch_size)
                meters['acc'].update(this_acc, batch_size)
            
            if game == 'concept':
                games_per_batch = 10
                sampled_img, sampled_y = sample_ref_game(img, y, games_per_batch = games_per_batch)
                for game in range(games_per_batch):
                    lis_scores = listener(torch.squeeze(sampled_img[:,game]), lang, lang_length)

                    # Evaluate loss and accuracy
                    y = torch.squeeze(sampled_y[:,game]).argmax(1)
                    this_loss = loss(lis_scores, y)
                    lis_pred = lis_scores.argmax(1)
                    this_acc = (lis_pred == y).float().mean().item()
                    
                    if training:
                        # SGD step
                        this_loss.backward(retain_graph=True)
                        optimizer.step()

                    meters['loss'].update(this_loss, batch_size)
                    meters['acc'].update(this_acc, batch_size)

    metrics = compute_average_metrics(meters)
    lang_onehot = lang.argmax(2)
    return metrics


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='Train',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_file', default=None, type=str)
    parser.add_argument('--pretrain_data_file', default=None, type=str)
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--game', choices=['concept', 'reference'], default='reference', type=str)
    parser.add_argument('--data_type', choices=['single', 'spatial'], default='single', type=str)
    parser.add_argument('--stim_dim', default=16, type=int)
    parser.add_argument('--stim_rep_dim', default=16, type=int)
    parser.add_argument('--vocab_size', default=None, type=int, help='Communication vocab size (default is number of shapes + colors)')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--train_size', default=1000, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--lr', default=1e-3, type=int)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--n_workers', default=0, type=int)

    args = parser.parse_args()
    
    # Data
    if args.pretrain and args.pretrain_data_file == None:
        if args.data_type == 'single':
            args.pretrain_data_file = ['../data/single/reference-1000-6.npz','../data/single/reference-1000-7.npz','../data/single/reference-1000-8.npz','../data/single/reference-1000-9.npz','../data/single/reference-1000-10.npz']
        if args.data_type == 'spatial':
            args.pretrain_data_file = ['../data/spatial/reference-1000-6.npz','../data/spatial/reference-1000-7.npz','../data/spatial/reference-1000-8.npz','../data/spatial/reference-1000-9.npz','../data/spatial/reference-1000-10.npz']
    if args.data_file == None:
        if args.game == 'reference':
            if args.data_type == 'single':
                args.data_file = ['../data/single/reference-1000-1.npz','../data/single/reference-1000-2.npz','../data/single/reference-1000-3.npz','../data/single/reference-1000-4.npz','../data/single/reference-1000-5.npz']
            if args.data_type == 'spatial':
                args.data_file = ['../data/spatial/reference-1000-1.npz','../data/spatial/reference-1000-2.npz','../data/spatial/reference-1000-3.npz','../data/spatial/reference-1000-4.npz','../data/spatial/reference-1000-5.npz']
        if args.game == 'concept':
            if args.data_type == 'single':
                args.data_file = ['../data/single/concept-1000-1.npz','../data/single/concept-1000-2.npz','../data/single/concept-1000-3.npz','../data/single/concept-1000-4.npz','../data/single/concept-1000-5.npz']
            if args.data_type == 'spatial':
                args.data_file = ['../data/spatial/concept-1000-1.npz','../data/spatial/concept-1000-2.npz','../data/spatial/concept-1000-3.npz','../data/spatial/concept-1000-4.npz','../data/spatial/concept-1000-5.npz']

    # Vocab
    # sos, eos, pad + n_vocab
    if args.vocab_size is not None:
        speaker_embs = nn.Embedding(4 + args.vocab_size, 50)
        listener_embs = nn.Embedding(4 + args.vocab_size, 50)
    else:
        speaker_embs = nn.Embedding(4 + len(VOCAB), 50)
        listener_embs = nn.Embedding(4 + len(VOCAB), 50)
    langs = np.array([])
    if args.pretrain:
        for file in args.pretrain_data_file:
            d = data.load_raw_data(file)
            langs = np.append(langs, d['langs'])
    for file in args.data_file:
        d = data.load_raw_data(file)
        langs = np.append(langs, d['langs'])
    vocab = data.init_vocab(langs)

    # Model
    speaker_vision = vision.Conv4()
    speaker = models.Speaker(speaker_vision, speaker_embs, args.game)
    listener_vision = vision.Conv4()
    listener = models.Listener(listener_vision, listener_embs)

    if args.cuda:
        speaker = speaker.cuda()
        listener = listener.cuda()

    # Optimization
    optimizer = optim.Adam(list(speaker.parameters()) +
                           list(listener.parameters()),
                           lr=args.lr)
    loss = nn.CrossEntropyLoss()

    # Metrics
    metrics = init_metrics()

    # Pretrain
    if args.pretrain:
        for epoch in range(args.epochs):
            train_metrics = defaultdict(list)
            train_metrics['loss'] = []
            train_metrics['acc'] = []
            val_metrics = defaultdict(list)
            val_metrics['loss'] = []
            val_metrics['acc'] = []
            for file in args.pretrain_data_file[0:len(args.pretrain_data_file)-1]:
                d = data.load_raw_data(file)
                dataloaders = {'train': DataLoader(ShapeWorld(d, vocab), batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)}
                run_args = (listener, optimizer, loss, dataloaders)
                temp_metrics = pretrain('train', epoch, *run_args)
                train_metrics['loss'].append(temp_metrics['loss'])
                train_metrics['acc'].append(temp_metrics['acc'])
            for file in [args.pretrain_data_file[-1]]:
                d = data.load_raw_data(file)
                dataloaders = {'val': DataLoader(ShapeWorld(d, vocab), batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)}
                run_args = (listener, optimizer, loss, dataloaders)
                temp_metrics = pretrain('val', epoch, *run_args)
                val_metrics['loss'].append(temp_metrics['loss'])
                val_metrics['acc'].append(temp_metrics['acc'])
            train_metrics['loss'] = np.mean(train_metrics['loss'])
            train_metrics['acc'] = np.mean(train_metrics['acc'])
            val_metrics['loss'] = np.mean(val_metrics['loss'])
            val_metrics['acc'] = np.mean(val_metrics['acc'])
            
            # Update your metrics, prepending the split name.
            for metric, value in train_metrics.items():
                metrics['train_{}'.format(metric)].append(value)
            for metric, value in val_metrics.items():
                metrics['val_{}'.format(metric)].append(value)
            metrics['current_epoch'] = epoch

            # Use validation accuracy to choose the best model. If it's the best,
            # update the best metrics.
            is_best = val_metrics['acc'] > metrics['best_acc']
            if is_best:
                metrics['best_acc'] = val_metrics['acc']
                metrics['best_loss'] = val_metrics['loss']
                metrics['best_epoch'] = epoch
                best_listener = copy.deepcopy(listener)
            print('Epoch '+str(epoch))
        listener = best_listener
        torch.save(listener, 'pretrained_listener.pt')
        print(metrics)
    
    for epoch in range(args.epochs):
        train_metrics = defaultdict(list)
        train_metrics['loss'] = []
        train_metrics['acc'] = []
        val_metrics = defaultdict(list)
        val_metrics['loss'] = []
        val_metrics['acc'] = []
        for file in args.data_file:
            d = data.load_raw_data(file)
            dataloaders = {'train': DataLoader(ShapeWorld(d, vocab), batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)}
            run_args = (speaker, listener, optimizer, loss, dataloaders, args.pretrain, args.game)
            temp_metrics = run('train', epoch, *run_args)
            train_metrics['loss'].append(temp_metrics['loss'])
            train_metrics['acc'].append(temp_metrics['acc'])
        for file in [args.data_file[-1]]:
            d = data.load_raw_data(file)
            dataloaders = {'val': DataLoader(ShapeWorld(d, vocab), batch_size=args.batch_size, shuffle=True,num_workers=args.n_workers)}
            run_args = (speaker, listener, optimizer, loss, dataloaders, args.pretrain, args.game)
            temp_metrics = run('val', epoch, *run_args)
            val_metrics['loss'].append(temp_metrics['loss'])
            val_metrics['acc'].append(temp_metrics['acc'])
        train_metrics['loss'] = np.mean(train_metrics['loss'])
        train_metrics['acc'] = np.mean(train_metrics['acc'])
        val_metrics['loss'] = np.mean(val_metrics['loss'])
        val_metrics['acc'] = np.mean(val_metrics['acc'])

        # Update your metrics, prepending the split name.
        for metric, value in train_metrics.items():
            metrics['train_{}'.format(metric)].append(value)
        for metric, value in val_metrics.items():
            metrics['val_{}'.format(metric)].append(value)
        metrics['current_epoch'] = epoch

        # Use validation accuracy to choose the best model. If it's the best,
        # update the best metrics.
        is_best = val_metrics['acc'] > metrics['best_acc']
        if is_best:
            metrics['best_acc'] = val_metrics['acc']
            metrics['best_loss'] = val_metrics['loss']
            metrics['best_epoch'] = epoch
            best_speaker = copy.deepcopy(speaker)
            best_listener = copy.deepcopy(listener)
        print('Epoch '+str(epoch))
    print(metrics)
