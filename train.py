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

def rsarun(split,
        epoch,
        speaker,
        optimizer,
        loss,
        dataloaders,
        num_samples=0,
        random_state=None):
    """
    Run the RSA model for a single epoch.

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
        context = contextlib.suppress()
    else:
        speaker.eval()
        context = torch.no_grad()  # Do not evaluate gradients for efficiency

    # Initialize your average meters to keep track of the epoch's running average
    measures = ['loss', 'acc']
    meters = {m: util.AverageMeter() for m in measures}

    sample_index = np.random.randint(0,len(dataloader),int(num_samples))
    with context:
        outputs = []
        for batch_i, (img, y, lang) in enumerate(dataloader):
            y = y.argmax(1) # convert from onehot
            batch_size = img.shape[0]

            # Convert to float
            img = img.float()
            if args.cuda:
                img = img.cuda()
                y = y.cuda()
                lang = lang.cuda()

            # Refresh the optimizer
            if training:
                optimizer.zero_grad()

            # Format lang
            max_len = 20
            length = torch.tensor([np.count_nonzero(t) for t in lang.cpu()], dtype=np.int)
            lang = F.one_hot(lang, num_classes=4+len(VOCAB))
            lang = F.pad(lang,(0,0,0,max_len-lang.shape[1])).float()
            
            # Forward pass
            hypo_out = speaker(img, lang, length, y)
            hint_seq = lang
            hypo_out = hypo_out[:, :-1].contiguous()
            hint_seq = hint_seq[:, 1:].contiguous()
            
            seq_len = hypo_out.size(1)
            
            hypo_out_2d = hypo_out.view(batch_size * seq_len, 4+len(VOCAB))
            hint_seq_2d = hint_seq.long().view(batch_size * seq_len, 4+len(VOCAB))
            hypo_loss = F.cross_entropy(hypo_out_2d, torch.max(hint_seq_2d, 1)[1], reduction='none')
            hypo_loss = hypo_loss.view(batch_size, seq_len)
            hypo_loss = torch.mean(torch.sum(hypo_loss, dim=1))

            if training:
                # SGD step
                hypo_loss.backward()
                optimizer.step()

            meters['loss'].update(hypo_loss, batch_size)
            meters['acc'].update(hypo_loss, batch_size)
            
            if batch_i in sample_index:
                print(img.shape)
                img = img[:,0]
                outputs.append([correct, 
                                np.array(dataloader.dataset.to_text(lang.argmax(2))), 
                                (img - torch.min(img))/(torch.max(img)-torch.min(img))])
                    
    metrics = compute_average_metrics(meters)
    return metrics, np.array(outputs)

def run(split,
        epoch,
        speaker,
        listener,
        optimizer,
        loss,
        dataloaders,
        pretrain,
        game_type,
        num_samples=0,
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

    sample_index = np.random.randint(0,len(dataloader),int(num_samples))
    with context:
        outputs = []
        for batch_i, (img, y, lang) in enumerate(dataloader):
            if game_type == 'reference':
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

            if game_type == 'reference':
                lis_scores = listener(img, lang, lang_length)
                
                # Evaluate loss and accuracy
                this_loss = loss(lis_scores, y.long())
                lis_pred = lis_scores.argmax(1)
                correct = (lis_pred == y)
                this_acc = correct.float().mean().item()
                
                if training:
                    # SGD step
                    this_loss.backward()
                    optimizer.step()

                meters['loss'].update(this_loss, batch_size)
                meters['acc'].update(this_acc, batch_size)
                
                if batch_i in sample_index:
                    outputs.append([correct, 
                                    np.array(dataloader.dataset.to_text(lang.argmax(2))), 
                                    (img - torch.min(img))/(torch.max(img)-torch.min(img))])
            if game_type == 'concept':
                games_per_batch = 5
                sampled_img, sampled_y = sample_ref_game(img, y, games_per_batch = games_per_batch)
                output = []
                for game in range(games_per_batch):
                    lis_scores = listener(torch.squeeze(sampled_img[:,game]), lang, lang_length)
                    # Evaluate loss and accuracy
                    y = torch.squeeze(sampled_y[:,game]).argmax(1)
                    this_loss = loss(lis_scores, y)
                    lis_pred = lis_scores.argmax(1)
                    correct = (lis_pred == y)
                    this_acc = correct.float().mean().item()
                    if training:
                        # SGD step
                        this_loss.backward(retain_graph=True)
                        optimizer.step()
                    meters['loss'].update(this_loss, batch_size)
                    meters['acc'].update(this_acc, batch_size)   
                    
                    if batch_i in sample_index:
                        output.append([correct, 
                                       np.array(dataloader.dataset.to_text(lang.argmax(2))), 
                                       (torch.squeeze(sampled_img[:,game]) - torch.min(torch.squeeze(sampled_img[:,game])))/(torch.max(torch.squeeze(sampled_img[:,game]))-torch.min(torch.squeeze(sampled_img[:,game])))])
                if batch_i in sample_index:
                    outputs.append(output)
                    
    metrics = compute_average_metrics(meters)
    return metrics, np.array(outputs)


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='Train',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_file', default=None, type=str)
    parser.add_argument('--pretrain_data_file', default=None, type=str)
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--new', action='store_true')
    parser.add_argument('--game_type', choices=['concept', 'reference'], default='reference', type=str)
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
    parser.add_argument('--rsa', action='store_true')

    args = parser.parse_args()
    
    # Data
    if args.pretrain and args.pretrain_data_file == None:
        if args.data_type == 'single':
            args.pretrain_data_file = ['../data/single/reference-1000-6.npz','../data/single/reference-1000-7.npz','../data/single/reference-1000-8.npz','../data/single/reference-1000-9.npz','../data/single/reference-1000-10.npz']
        if args.data_type == 'spatial':
            args.pretrain_data_file = ['../data/spatial/reference-1000-6.npz','../data/spatial/reference-1000-7.npz','../data/spatial/reference-1000-8.npz','../data/spatial/reference-1000-9.npz','../data/spatial/reference-1000-10.npz']
    if args.data_file == None:
        if args.game_type == 'reference':
            if args.data_type == 'single':
                args.data_file = ['../data/single/reference-1000-1.npz','../data/single/reference-1000-2.npz','../data/single/reference-1000-3.npz','../data/single/reference-1000-4.npz','../data/single/reference-1000-5.npz']
            if args.data_type == 'spatial':
                args.data_file = ['../data/spatial/reference-1000-1.npz','../data/spatial/reference-1000-2.npz','../data/spatial/reference-1000-3.npz','../data/spatial/reference-1000-4.npz','../data/spatial/reference-1000-5.npz']
        if args.game_type == 'concept':
            if args.data_type == 'single':
                args.data_file = ['../data/single/concept-100-1.npz','../data/single/concept-100-2.npz','../data/single/concept-100-3.npz','../data/single/concept-100-4.npz','../data/single/concept-100-5.npz','../data/single/concept-100-6.npz','../data/single/concept-100-7.npz','../data/single/concept-100-8.npz','../data/single/concept-100-9.npz','../data/single/concept-100-10.npz','../data/single/concept-100-11.npz','../data/single/concept-100-12.npz','../data/single/concept-100-13.npz','../data/single/concept-100-14.npz','../data/single/concept-100-15.npz','../data/single/concept-100-16.npz','../data/single/concept-100-17.npz','../data/single/concept-100-18.npz','../data/single/concept-100-19.npz','../data/single/concept-100-20.npz','../data/single/concept-100-21.npz','../data/single/concept-100-22.npz','../data/single/concept-100-23.npz','../data/single/concept-100-24.npz','../data/single/concept-100-25.npz','../data/single/concept-100-26.npz','../data/single/concept-100-27.npz','../data/single/concept-100-28.npz','../data/single/concept-100-29.npz','../data/single/concept-100-30.npz','../data/single/concept-100-31.npz','../data/single/concept-100-32.npz','../data/single/concept-100-33.npz','../data/single/concept-100-34.npz','../data/single/concept-100-35.npz','../data/single/concept-100-36.npz','../data/single/concept-100-37.npz','../data/single/concept-100-38.npz','../data/single/concept-100-39.npz','../data/single/concept-100-40.npz','../data/single/concept-100-41.npz','../data/single/concept-100-42.npz','../data/single/concept-100-43.npz','../data/single/concept-100-44.npz','../data/single/concept-100-45.npz','../data/single/concept-100-46.npz','../data/single/concept-100-47.npz','../data/single/concept-100-48.npz','../data/single/concept-100-49.npz','../data/single/concept-100-50.npz']
            if args.data_type == 'spatial':
                args.data_file = ['../data/spatial/concept-1000-1.npz','../data/spatial/concept-1000-2.npz','../data/spatial/concept-1000-3.npz','../data/spatial/concept-1000-4.npz','../data/spatial/concept-1000-5.npz']

    # Vocab
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

    if args.rsa:
        # Model
        speaker_vision = vision.Conv4()
        speaker = models.LiteralSpeaker(speaker_vision, speaker_embs)
        if args.cuda:
            speaker = speaker.cuda()
        
        # Optimization
        optimizer = optim.Adam(list(speaker.parameters()),
                               lr=args.lr)
        loss = nn.CrossEntropyLoss()
        
        # Metrics
        metrics = init_metrics()
        
        for epoch in range(args.epochs):
            train_metrics = defaultdict(list)
            train_metrics['loss'] = []
            train_metrics['acc'] = []
            val_metrics = defaultdict(list)
            val_metrics['loss'] = []
            val_metrics['acc'] = []
            sample_index = np.random.randint(0,len(args.data_file)-1,3)
            sample_file = [args.data_file[index] for index in sample_index]
            outputs = []
            for file in args.data_file[0:len(args.data_file)-1]:
                d = data.load_raw_data(file)
                dataloaders = {'train': DataLoader(ShapeWorld(d, vocab), batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)}
                run_args = (speaker, optimizer, loss, dataloaders)
                if epoch == args.epochs-1 and file in sample_file:
                    temp_metrics, output = rsarun('train', epoch, *run_args, 1)
                    outputs.append(output)
                else:
                    temp_metrics, output = rsarun('train', epoch, *run_args)
                train_metrics['loss'].append(temp_metrics['loss'])
                train_metrics['acc'].append(temp_metrics['acc'])
            if epoch == args.epochs-1:
                outputs = np.array(outputs)
                outputs = outputs.reshape(outputs.shape[0]*outputs.shape[1],outputs.shape[2])
                for output_index, output in enumerate(outputs):
                    np.savetxt('../output/literal_speaker/train/correct_'+str(output_index)+'.txt',output[0].cpu().numpy())
                    np.savetxt('../output/literal_speaker/train/lang_'+str(output_index)+'.txt',output[1],fmt='%s')
                    for batch_index, batch in enumerate(output[2]):
                            plt.imsave('../output/literal_speaker/train/img_'+str(output_index)+'_'+str(batch_index)+'_1.png',batch[0].permute(1,2,0).cpu().numpy())
                            plt.imsave('../output/literal_speaker/train/img_'+str(output_index)+'_'+str(batch_index)+'_2.png',batch[1].permute(1,2,0).cpu().numpy())
                            plt.imsave('../output/literal_speaker/train/img_'+str(output_index)+'_'+str(batch_index)+'_3.png',batch[2].permute(1,2,0).cpu().numpy())
            outputs = []
            for file in [args.data_file[-1]]:
                d = data.load_raw_data(file)
                dataloaders = {'val': DataLoader(ShapeWorld(d, vocab), batch_size=args.batch_size, shuffle=True,num_workers=args.n_workers)}
                run_args = (speaker, optimizer, loss, dataloaders)
                if epoch == args.epochs-1:
                    temp_metrics, output = rsarun('val', epoch, *run_args, 1)
                    outputs.append(output)
                else:
                    temp_metrics, output = rsarun('val', epoch, *run_args)
                val_metrics['loss'].append(temp_metrics['loss'])
                val_metrics['acc'].append(temp_metrics['acc'])
            if epoch == args.epochs-1:
                outputs = np.array(outputs)
                outputs = outputs.reshape(outputs.shape[0]*outputs.shape[1],outputs.shape[2])
                for output_index, output in enumerate(outputs):
                    np.savetxt('../output/literal_speaker/val/correct_'+str(output_index)+'.txt',output[0].cpu().numpy())
                    np.savetxt('../output/literal_speaker/val/lang_'+str(output_index)+'.txt',output[1],fmt='%s')
                    for batch_index, batch in enumerate(output[2]):
                        plt.imsave('../output/literal_speaker/val/img_'+str(output_index)+'_'+str(batch_index)+'_1.png',batch[0].permute(1,2,0).cpu().numpy())
                        plt.imsave('../output/literal_speaker/val/img_'+str(output_index)+'_'+str(batch_index)+'_2.png',batch[1].permute(1,2,0).cpu().numpy())
                        plt.imsave('../output/literal_speaker/val/img_'+str(output_index)+'_'+str(batch_index)+'_3.png',batch[2].permute(1,2,0).cpu().numpy())
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
            print('Epoch '+str(epoch))
            print(train_metrics['acc'])
            print(val_metrics['acc'])
        print(metrics)
    else:
        # Model
        speaker_vision = vision.Conv4()
        speaker = models.Speaker(speaker_vision, speaker_embs, args.game_type)
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
            if args.new:
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
                torch.save(vocab, 'vocab.pt')
                print(metrics)
            else:
                listener = torch.load('pretrained_listener.pt')
                vocab = torch.load('vocab.pt')
        
        # Run
        for epoch in range(args.epochs):
            train_metrics = defaultdict(list)
            train_metrics['loss'] = []
            train_metrics['acc'] = []
            val_metrics = defaultdict(list)
            val_metrics['loss'] = []
            val_metrics['acc'] = []
            sample_index = np.random.randint(0,len(args.data_file)-1,3)
            sample_file = [args.data_file[index] for index in sample_index]
            outputs = []
            for file in args.data_file[0:len(args.data_file)-1]:
                d = data.load_raw_data(file)
                dataloaders = {'train': DataLoader(ShapeWorld(d, vocab), batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)}
                run_args = (speaker, listener, optimizer, loss, dataloaders, args.pretrain, args.game_type)
                if epoch == args.epochs-1 and file in sample_file:
                    temp_metrics, output = run('train', epoch, *run_args, 1)
                    outputs.append(output)
                else:
                    temp_metrics, output = run('train', epoch, *run_args)
                train_metrics['loss'].append(temp_metrics['loss'])
                train_metrics['acc'].append(temp_metrics['acc'])
            if epoch == args.epochs-1:
                if args.game_type == 'concept':
                    outputs = np.array(outputs)
                    outputs = outputs.reshape(outputs.shape[0]*outputs.shape[1],outputs.shape[2],outputs.shape[3])
                    for output_index, output in enumerate(outputs):
                        for game_index, game in enumerate(output):
                            if args.pretrain:
                                np.savetxt('../output/pretrain_concept/train/correct_'+str(output_index)+'_'+str(game_index)+'.txt',game[0].cpu().numpy())
                                np.savetxt('../output/pretrain_concept/train/lang_'+str(output_index)+'_'+str(game_index)+'.txt',game[1],fmt='%s')
                                for batch_index, batch in enumerate(game[2]):
                                    plt.imsave('../output/pretrain_concept/train/img_'+str(output_index)+'_'+str(batch_index)+'_'+str(game_index)+'_1.png',batch[0].permute(1,2,0).cpu().numpy())
                                    plt.imsave('../output/pretrain_concept/train/img_'+str(output_index)+'_'+str(batch_index)+'_'+str(game_index)+'_2.png',batch[1].permute(1,2,0).cpu().numpy())
                                    plt.imsave('../output/pretrain_concept/train/img_'+str(output_index)+'_'+str(batch_index)+'_'+str(game_index)+'_3.png',batch[2].permute(1,2,0).cpu().numpy())
                            else:
                                np.savetxt('../output/concept/train/correct_'+str(output_index)+'_'+str(game_index)+'.txt',game[0].cpu().numpy())
                                np.savetxt('../output/concept/train/lang_'+str(output_index)+'_'+str(game_index)+'.txt',game[1],fmt='%s')
                                for batch_index, batch in enumerate(game[2]):
                                    plt.imsave('../output/concept/train/img_'+str(output_index)+'_'+str(batch_index)+'_'+str(game_index)+'_1.png',batch[0].permute(1,2,0).cpu().numpy())
                                    plt.imsave('../output/concept/train/img_'+str(output_index)+'_'+str(batch_index)+'_'+str(game_index)+'_2.png',batch[1].permute(1,2,0).cpu().numpy())
                                    plt.imsave('../output/concept/train/img_'+str(output_index)+'_'+str(batch_index)+'_'+str(game_index)+'_3.png',batch[2].permute(1,2,0).cpu().numpy())
                else:
                    outputs = np.array(outputs)
                    outputs = outputs.reshape(outputs.shape[0]*outputs.shape[1],outputs.shape[2])
                    for output_index, output in enumerate(outputs):
                        if args.pretrain:
                            np.savetxt('../output/pretrain_reference/train/correct_'+str(output_index)+'.txt',output[0].cpu().numpy())
                            np.savetxt('../output/pretrain_reference/train/lang_'+str(output_index)+'.txt',output[1],fmt='%s')
                            for batch_index, batch in enumerate(output[2]):
                                    plt.imsave('../output/pretrain_reference/train/img_'+str(output_index)+'_'+str(batch_index)+'_1.png',batch[0].permute(1,2,0).cpu().numpy())
                                    plt.imsave('../output/pretrain_reference/train/img_'+str(output_index)+'_'+str(batch_index)+'_2.png',batch[1].permute(1,2,0).cpu().numpy())
                                    plt.imsave('../output/pretrain_reference/train/img_'+str(output_index)+'_'+str(batch_index)+'_3.png',batch[2].permute(1,2,0).cpu().numpy())
                        else:
                            np.savetxt('../output/reference/train/correct_'+str(output_index)+'.txt',output[0].cpu().numpy())
                            np.savetxt('../output/reference/train/lang_'+str(output_index)+'.txt',output[1],fmt='%s')
                            for batch_index, batch in enumerate(output[2]):
                                plt.imsave('../output/reference/train/img_'+str(output_index)+'_'+str(batch_index)+'_1.png',batch[0].permute(1,2,0).cpu().numpy())
                                plt.imsave('../output/reference/train/img_'+str(output_index)+'_'+str(batch_index)+'_2.png',batch[1].permute(1,2,0).cpu().numpy())
                                plt.imsave('../output/reference/train/img_'+str(output_index)+'_'+str(batch_index)+'_3.png',batch[2].permute(1,2,0).cpu().numpy())
            outputs = []
            for file in [args.data_file[-1]]:
                d = data.load_raw_data(file)
                dataloaders = {'val': DataLoader(ShapeWorld(d, vocab), batch_size=args.batch_size, shuffle=True,num_workers=args.n_workers)}
                run_args = (speaker, listener, optimizer, loss, dataloaders, args.pretrain, args.game_type)
                if epoch == args.epochs-1:
                    temp_metrics, output = run('val', epoch, *run_args, 1)
                    outputs.append(output)
                else:
                    temp_metrics, output = run('val', epoch, *run_args)
                val_metrics['loss'].append(temp_metrics['loss'])
                val_metrics['acc'].append(temp_metrics['acc'])
            if epoch == args.epochs-1:
                if args.game_type == 'concept':
                    outputs = np.array(outputs)
                    outputs = outputs.reshape(outputs.shape[0]*outputs.shape[1],outputs.shape[2],outputs.shape[3])
                    for output_index, output in enumerate(outputs):
                        for game_index, game in enumerate(output):
                            if args.pretrain:
                                np.savetxt('../output/pretrain_concept/val/correct_'+str(output_index)+'_'+str(game_index)+'.txt',game[0].cpu().numpy())
                                np.savetxt('../output/pretrain_concept/val/lang_'+str(output_index)+'_'+str(game_index)+'.txt',game[1],fmt='%s')
                                for batch in game[2]:
                                    plt.imsave('../output/pretrain_concept/val/img_'+str(output_index)+'_'+str(game_index)+'_1.png',batch[0].permute(1,2,0).cpu().numpy())
                                    plt.imsave('../output/pretrain_concept/val/img_'+str(output_index)+'_'+str(game_index)+'_2.png',batch[1].permute(1,2,0).cpu().numpy())
                                    plt.imsave('../output/pretrain_concept/val/img_'+str(output_index)+'_'+str(game_index)+'_3.png',batch[2].permute(1,2,0).cpu().numpy())
                            else:
                                np.savetxt('../output/concept/val/correct_'+str(output_index)+'_'+str(game_index)+'.txt',game[0].cpu().numpy())
                                np.savetxt('../output/concept/val/lang_'+str(output_index)+'_'+str(game_index)+'.txt',game[1],fmt='%s')
                                for batch in game[2]:
                                    plt.imsave('../output/concept/val/img_'+str(output_index)+'_'+str(game_index)+'_1.png',batch[0].permute(1,2,0).cpu().numpy())
                                    plt.imsave('../output/concept/val/img_'+str(output_index)+'_'+str(game_index)+'_2.png',batch[1].permute(1,2,0).cpu().numpy())
                                    plt.imsave('../output/concept/val/img_'+str(output_index)+'_'+str(game_index)+'_3.png',batch[2].permute(1,2,0).cpu().numpy())
                else:
                    outputs = np.array(outputs)
                    outputs = outputs.reshape(outputs.shape[0]*outputs.shape[1],outputs.shape[2])
                    for output_index, output in enumerate(outputs):
                        if args.pretrain:
                            np.savetxt('../output/pretrain_reference/val/correct_'+str(output_index)+'.txt',output[0].cpu().numpy())
                            np.savetxt('../output/pretrain_reference/val/lang_'+str(output_index)+'.txt',output[1],fmt='%s')
                            for batch_index, batch in enumerate(output[2]):
                                    plt.imsave('../output/pretrain_reference/val/img_'+str(output_index)+'_'+str(batch_index)+'_1.png',batch[0].permute(1,2,0).cpu().numpy())
                                    plt.imsave('../output/pretrain_reference/val/img_'+str(output_index)+'_'+str(batch_index)+'_2.png',batch[1].permute(1,2,0).cpu().numpy())
                                    plt.imsave('../output/pretrain_reference/val/img_'+str(output_index)+'_'+str(batch_index)+'_3.png',batch[2].permute(1,2,0).cpu().numpy())
                        else:
                            np.savetxt('../output/reference/val/correct_'+str(output_index)+'.txt',output[0].cpu().numpy())
                            np.savetxt('../output/reference/val/lang_'+str(output_index)+'.txt',output[1],fmt='%s')
                            for batch_index, batch in enumerate(output[2]):
                                    plt.imsave('../output/reference/val/img_'+str(output_index)+'_'+str(batch_index)+'_1.png',batch[0].permute(1,2,0).cpu().numpy())
                                    plt.imsave('../output/reference/val/img_'+str(output_index)+'_'+str(batch_index)+'_2.png',batch[1].permute(1,2,0).cpu().numpy())
                                    plt.imsave('../output/reference/val/img_'+str(output_index)+'_'+str(batch_index)+'_3.png',batch[2].permute(1,2,0).cpu().numpy())
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
            print(train_metrics['acc'])
            print(val_metrics['acc'])
        print(metrics)
