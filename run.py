import contextlib
import random
from collections import defaultdict
import copy
import math

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
from shapeworld import SHAPES, COLORS, VOCAB

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

def run(epoch, data_file, split, run_type, speaker, listener, optimizer, loss, vocab, batch_size, cuda, game_type = 'reference', num_samples = 5, get_outputs = False, srr = True):
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
    
    outputs = {'lang':[],'pred':[],'score':[], 'in':[]}
    
    loss2 = nn.CrossEntropyLoss(reduction="none")
    
    language_model = torch.load('language-model.pt')
    language_model.eval()
    
    if run_type == 'sample':
        listener2 = torch.load('pretrained-listener-01.pt')
    
    if split == 'train':
        if run_type == 'literal' or run_type == 'lm':
            speaker.train()
        elif run_type == 'pretrain':
            listener.train()
        else:
            speaker.train()
            listener.train()
        context = contextlib.suppress()
    else:
        if run_type != 'pretrain':
            speaker.eval()
        if run_type != 'literal' and run_type != 'lm':
            listener.eval()
        context = torch.no_grad()  # Do not evaluate gradients for efficiency

    # Initialize your average meters to keep track of the epoch's running average
    if get_outputs:
        ci_listeners = [torch.load('pretrained-listener-2.pt'),
                        torch.load('pretrained-listener-3.pt'),
                        torch.load('pretrained-listener-4.pt'),
                        torch.load('pretrained-listener-5.pt'),
                        torch.load('pretrained-listener-6.pt'),
                        torch.load('pretrained-listener-7.pt'),
                        torch.load('pretrained-listener-8.pt'),
                        torch.load('pretrained-listener-9.pt'),
                        torch.load('pretrained-listener-10.pt')]
        for ci_listener in ci_listeners:
            ci_listener.eval()
        meters = {'loss':[], 'acc':[], 'prob':[], 'perp':[], 'CI low':[], 'CI high':[]}
    else:
        measures = ['loss', 'acc']
        meters = {m: util.AverageMeter() for m in measures}

    with context:
        for file in data_file:
            d = data.load_raw_data(file)
            if get_outputs:
                dataloader = DataLoader(ShapeWorld(d, vocab), batch_size=batch_size, shuffle=False)
            else:
                dataloader = DataLoader(ShapeWorld(d, vocab), batch_size=batch_size, shuffle=True)
            for batch_i, (img, y, lang) in enumerate(dataloader):
                if get_outputs:
                    outputs['in'].append(lang)
                if run_type == 'literal' or run_type == 'pretrain' or game_type == 'reference' or run_type == 'sample':
                    y = y.argmax(1) # convert from onehot
                batch_size = img.shape[0]

                # Convert to float
                img = img.float()

                # Refresh the optimizer
                if split == 'train':
                    optimizer.zero_grad()

                if run_type == 'literal' or run_type == 'pretrain' or run_type == 'lm':
                    max_len = 20
                    length = torch.tensor([np.count_nonzero(t) for t in lang.cpu()], dtype=np.int)
                    lang = F.one_hot(lang, num_classes = 4+len(VOCAB))
                    lang = F.pad(lang,(0,0,0,max_len-lang.shape[1])).float()
                    for B in range(lang.shape[0]):
                        for L in range(lang.shape[1]):
                            if lang[B][L].sum() == 0:
                                lang[B][L][0] = 1

                if cuda:
                    img = img.cuda()
                    y = y.cuda()
                    lang.cuda()
                    if run_type == 'literal' or run_type == 'pretrain' or run_type == 'lm':
                        length = length.cuda()
                
                # Forward pass
                if run_type == 'pretrain':
                    lis_scores = listener(img, lang, length)
                elif run_type == 'literal':
                    lang_out = speaker(img, lang, length, y)
                elif run_type == 'lm':
                    lang_out = speaker(lang, length)
                elif run_type == 'sample':
                    if srr:
                        langs, lang_lengths = speaker.sample(img, y, greedy = True)
                    else:
                        langs, lang_lengths, eos_loss = speaker(img, y)
                    langs = langs.unsqueeze(0); lang_lengths = lang_lengths.unsqueeze(0)
                    for _ in range(num_samples-1):
                        if srr:
                            lang, lang_length = speaker.sample(img, y)
                        else:
                            lang, lang_length, eos_loss = speaker(img, y)
                        lang = lang.unsqueeze(0); lang_length = lang_length.unsqueeze(0)
                        langs = torch.cat((langs, lang), 0)
                        lang_lengths = torch.cat((lang_lengths, lang_length), 0)
                    lang = langs[:,0]
                else:
                    lang, lang_length, eos_loss = speaker(img, y)

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
                elif run_type == 'literal' or run_type == 'lm':
                    lang_out = lang_out[:, :-1].contiguous()
                    lang = lang[:, 1:].contiguous()
                    lang_out = lang_out.view(batch_size*lang_out.size(1), 4+len(VOCAB))
                    lang = lang.long().view(batch_size*lang_out.size(1), 4+len(VOCAB))
                    this_loss = loss(lang_out.cuda(), torch.max(lang, 1)[1])

                    if split == 'train':
                        # SGD step
                        this_loss.backward()
                        optimizer.step()
                        
                    this_acc = (lang_out.argmax(1)==lang.argmax(1)).float().mean().item()

                    meters['loss'].update(this_loss, batch_size)
                    meters['acc'].update(this_acc, batch_size)
                else:
                    if game_type == 'reference':
                        if run_type == 'sample':
                            best_lis_scores = torch.zeros((batch_size,3))
                            best_lis_pred = torch.zeros(batch_size)
                            best_correct = torch.zeros(batch_size)
                            best_this_acc = -math.inf*torch.ones(batch_size)
                            best_lang = torch.zeros((langs.shape[1],langs.shape[2],langs.shape[3]))
                            best_lang_length = torch.zeros(lang_lengths.shape[1])
                            for lang, lang_length in zip(langs, lang_lengths):
                                lis_scores = listener(img, lang, lang_length)
                                lis_pred = lis_scores.argmax(1)
                                correct = (lis_pred == y)
                                for game in range(batch_size):
                                    this_acc = correct[game].float().mean().item()
                                    if this_acc>best_this_acc[game]:
                                        best_lis_scores[game] = lis_scores[game]
                                        best_lis_pred[game] = lis_pred[game]
                                        best_correct[game] = correct[game]
                                        best_this_acc[game] = this_acc
                                        best_lang[game] = lang[game]
                                        best_lang_length[game] = lang_length[game]
                            
                            lang = best_lang
                            lang_length = best_lang_length
                            lis_scores = listener2(img, lang, lang_length)
                            
                            # Evaluate loss and accuracy
                            lis_pred = lis_scores.argmax(1)
                            correct = (lis_pred == y)
                            this_acc = correct.float().mean().item()
                            this_loss = loss(lis_scores.cuda(), y.long())
                            this_acc = correct.float().mean().item()
                            
                            if split == 'train':
                                # SGD step
                                this_loss.backward()
                                optimizer.step()
                            
                            if get_outputs:
                                """
                                seq_prob = language_model.probability(lang,lang_length)
                                seq_perp = []
                                for i, prob in enumerate(seq_prob):
                                    seq_prob[i] = math.e**prob
                                    seq_perp.append(2**(-1/lang_length[i]*math.log(seq_prob[i],2)))
                                seq_prob = seq_prob.mean()
                                seq_perp = torch.tensor(seq_perp).mean()
                                """
                                lm_seq = language_model(lang,lang_length)[:, :-1].contiguous()
                                temp_prob = loss2(lm_seq.view(batch_size*lm_seq.size(1),4+len(VOCAB)).cuda(),torch.max(lang.cuda()[:,1:].contiguous().long().view(batch_size*lm_seq.size(1),4+len(VOCAB)),1)[1])
                                print(temp_prob)
                                temp_prob = temp_prob.view(batch_size,lm_seq.size(1))
                                seq_prob = []
                                for i,prob in enumerate(temp_prob):
                                    seq_prob.append(1)
                                    for j in range(lang_length[i].int()-1):
                                        seq_prob[-1] += prob[j]
                                seq_prob = torch.tensor(seq_prob).mean().numpy()
                                seq_perp = math.exp(-seq_prob)
                                
                                low_acc = 1
                                high_acc = 0
                                for ci_listener in ci_listeners:
                                    correct = (ci_listener(img,lang,lang_length).argmax(1) == y)
                                    acc = correct.float().mean().item()
                                    if low_acc>acc:
                                        low_acc = acc
                                    if high_acc<acc:
                                        high_acc = acc
                                this_loss = this_loss.cpu().numpy()
                                
                                outputs['lang'].append(lang.argmax(2))
                                outputs['pred'].append(lis_pred)
                                outputs['score'].append(lis_scores)
                                meters['loss'].append(this_loss)
                                meters['acc'].append(this_acc)
                                meters['prob'].append(seq_prob)
                                meters['perp'].append(seq_perp)
                                meters['CI low'].append(low_acc)
                                meters['CI high'].append(high_acc)
                            else:
                                meters['loss'].update(this_loss, batch_size)
                                meters['acc'].update(this_acc, batch_size)
                        else:
                            lis_scores = listener(img, lang, lang_length)
                            
                            # Evaluate loss and accuracy
                            if run_type == 'pretrain':
                                this_loss = loss(lis_scores, y.long()) #+prob*0.5 #+seq_prob.detach()*0.005 #+eos_prob*0.01
                            else:
                                this_loss = loss(lis_scores, y.long())
                            lis_pred = lis_scores.argmax(1)
                            correct = (lis_pred == y)
                            this_acc = correct.float().mean().item()
                            
                            if split == 'train':
                                # SGD step
                                this_loss.backward()
                                optimizer.step()
                             
                            if get_outputs:
                                """
                                seq_prob = language_model.probability(lang,lang_length)
                                seq_perp = []
                                for i, prob in enumerate(seq_prob):
                                    seq_prob[i] = math.e**prob
                                    seq_perp.append(2**(-1/lang_length[i]*math.log(seq_prob[i],2)))
                                seq_prob = seq_prob.mean()
                                seq_perp = torch.tensor(seq_perp).mean()
                                """
                                lm_seq = language_model(lang,lang_length)[:, :-1].contiguous()
                                temp_prob = loss2(lm_seq.view(batch_size*lm_seq.size(1),4+len(VOCAB)).cuda(),torch.max(lang.cuda()[:,1:].contiguous().long().view(batch_size*lm_seq.size(1),4+len(VOCAB)),1)[1])
                                temp_prob = temp_prob.view(batch_size,lm_seq.size(1))
                                seq_prob = []
                                for i,prob in enumerate(temp_prob):
                                    seq_prob.append(1)
                                    for j in range(lang_length[i].int()-1):
                                        seq_prob[-1] += prob[j]
                                seq_prob = torch.tensor(seq_prob).mean().numpy()
                                seq_perp = math.exp(-seq_prob)
                                
                                low_acc = 1
                                high_acc = 0
                                for ci_listener in ci_listeners:
                                    correct = (ci_listener(img,lang,lang_length).argmax(1) == y)
                                    acc = correct.float().mean().item()
                                    if low_acc>acc:
                                        low_acc = acc
                                    if high_acc<acc:
                                        high_acc = acc
                                this_loss = this_loss.cpu().numpy()
                                
                                outputs['lang'].append(lang.argmax(2))
                                outputs['pred'].append(lis_pred)
                                outputs['score'].append(lis_scores)
                                meters['loss'].append(this_loss)
                                meters['acc'].append(this_acc)
                                meters['prob'].append(seq_prob)
                                meters['perp'].append(seq_perp)
                                meters['CI low'].append(low_acc)
                                meters['CI high'].append(high_acc)
                            else:
                                meters['loss'].update(this_loss, batch_size) #-prob*0.5, batch_size) #-seq_prob.detach()*0.005, batch_size) #-eos_prob*0.01, batch_size)
                                meters['acc'].update(this_acc, batch_size)
                                
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
    if get_outputs:
        meters['loss'] = np.array(meters['loss']).tolist()
        meters['prob'] = np.array(meters['prob']).tolist()
        metrics = meters
    else:
        metrics = compute_average_metrics(meters)
    return metrics, outputs