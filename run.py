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

def collect_outputs(meters, outputs, vocab, img, y, lang, lang_length, lis_pred, lis_scores, this_loss, this_acc, batch_size, ci_listeners, language_model, loss2):
    lm_seq = language_model(lang,lang_length)[:, :-1].contiguous()
    temp_prob = loss2(lm_seq.view(batch_size*lm_seq.size(1),len(vocab['w2i'].keys())).cuda(),torch.max(lang.cuda()[:,1:].contiguous().long().view(batch_size*lm_seq.size(1),len(vocab['w2i'].keys())),1)[1])
    temp_prob = temp_prob.view(batch_size,lm_seq.size(1))
    seq_prob = []
    #seq_perp = []
    log_prob = []
    for i,prob in enumerate(temp_prob):
        log_prob.append(1)
        for j in range(lang_length[i].int()-1):
            log_prob[-1] += -prob[j]
        #seq_perp.append(math.exp(-log_prob[-1]))
        seq_prob.append(math.exp(log_prob[-1]))
    seq_prob = torch.tensor(seq_prob).mean().numpy()
    #seq_perp = torch.tensor(seq_perp).mean().numpy()
    log_prob = torch.tensor(log_prob).mean().numpy()

    if ci_listeners != None:
        ci = []
        for ci_listener in ci_listeners:
            correct = (ci_listener(img,lang,lang_length).argmax(1)==y)
            acc = correct.float().mean().item()
            ci.append(acc)
    
    lang = lang.argmax(2)
    outputs['lang'].append(lang)
    outputs['pred'].append(lis_pred)
    outputs['score'].append(lis_scores)
    meters['loss'].append(this_loss.cpu().numpy())
    meters['acc'].append(this_acc)
    meters['prob'].append(seq_prob)
    if ci_listeners != None:
        meters['CI'].append(ci)
    meters['length'].append(lang_length.float().mean().cpu().numpy())
    colors = 0
    for color in [4, 6, 9, 10, 11, 14]:
        colors += (lang == color).sum(dim=1).float().mean()
    shapes = 0
    for shape in [7, 8, 12, 13]:
        shapes += (lang == shape).sum(dim=1).float().mean()
    meters['colors'].append(colors.cpu().numpy())
    meters['shapes'].append(shapes.cpu().numpy())
    return meters, outputs

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

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def run(epoch, data_file, split, run_type, speaker, listener, optimizer, loss, vocab, batch_size, cuda, game_type = 'reference', num_samples = None, get_outputs = False, srr = True, lm_wt = None, test_type = None, activation = 'gumbel', ci = True, dataset = 'single', penalty = None, tau = 1):
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
    max_len = 40
    outputs = {'lang':[],'pred':[],'score':[], 'in':[]}
    
    loss2 = nn.CrossEntropyLoss(reduction="none")
    
    if run_type == 'sample' or run_type == 'rsa':
        if dataset == 'single':
            internal_listener = torch.load('./models/single/pretrained-listener-01.pt')
        else:
            internal_listener = torch.load('./models/'+dataset+'/pretrained-listener-0.pt')
    
    if split == 'train':
        language_model = torch.load('./models/'+dataset+'/language-model.pt')
        for param in language_model.parameters():
            param.requires_grad = False
        language_model.train()
        if run_type == 'literal' or run_type == 'lm':
            speaker.train()
        elif run_type == 'pretrain':
            listener.train()
        else:
            speaker.train()
            if run_type == 'pretrained':
                for param in listener.parameters():
                    param.requires_grad = False
            listener.train()
        context = contextlib.suppress()
    else:
        language_model = torch.load('./models/'+dataset+'/language-model.pt')
        language_model.eval()
        if run_type != 'pretrain' and run_type != 'oracle' and run_type != 'test':
            speaker.eval()
        if run_type != 'literal' and run_type != 'lm':
            listener.eval()
        context = torch.no_grad()  # Do not evaluate gradients for efficiency

    # Initialize your average meters to keep track of the epoch's running average
    if get_outputs:
        if ci == True:
            ci_listeners = [torch.load('./models/single/pretrained-listener-2.pt'),
                            torch.load('./models/single/pretrained-listener-3.pt'),
                            torch.load('./models/single/pretrained-listener-4.pt'),
                            torch.load('./models/single/pretrained-listener-5.pt'),
                            torch.load('./models/single/pretrained-listener-6.pt'),
                            torch.load('./models/single/pretrained-listener-7.pt'),
                            torch.load('./models/single/pretrained-listener-8.pt'),
                            torch.load('./models/single/pretrained-listener-9.pt'),
                            torch.load('./models/single/pretrained-listener-10.pt')]
            for ci_listener in ci_listeners:
                ci_listener.eval()
            meters = {'loss':[], 'acc':[], 'prob':[], 'CI':[], 'length':[], 'colors':[], 'shapes':[]}
        else:
            ci_listeners = None
            meters = {'loss':[], 'acc':[], 'prob':[], 'length':[], 'colors':[], 'shapes':[]}
        
    else:
        if run_type == 'literal' or run_type == 'lm' or run_type == 'pretrain':
            measures = ['loss', 'acc']
        else:
            measures = ['loss', 'lm loss', 'acc', 'length']
        meters = {m: util.AverageMeter() for m in measures}

    with context:
        for file in data_file:
            d = data.load_raw_data(file)
            if get_outputs:
                dataloader = DataLoader(ShapeWorld(d, vocab), batch_size=batch_size, shuffle=False)
            else:
                dataloader = DataLoader(ShapeWorld(d, vocab), batch_size=batch_size, shuffle=True)
                
            for batch_i, (img, y, lang) in enumerate(dataloader):
                true_lang = lang
                true_lang_length = torch.tensor([np.count_nonzero(t) for t in lang.cpu()], dtype=np.int)
                #img = img.flip(1)
                #y = y.flip(1)
                if get_outputs:
                    outputs['in'].append(lang)
                if run_type == 'literal' or run_type == 'pretrain' or game_type == 'reference' or run_type == 'sample' or run_type == 'rsa' or run_type == 'test':
                    y = y.argmax(1) # convert from onehot
                batch_size = img.shape[0]

                # Convert to float
                img = img.float()

                # Refresh the optimizer
                if split == 'train':
                    optimizer.zero_grad()

                if run_type == 'literal' or run_type == 'pretrain' or run_type == 'lm' or run_type == 'oracle':
                    max_len = 40
                    length = torch.tensor([np.count_nonzero(t) for t in lang.cpu()], dtype=np.int)
                    lang[lang>=len(vocab['w2i'].keys())] = 3
                    lang = F.one_hot(lang, num_classes = len(vocab['w2i'].keys()))
                    lang = F.pad(lang,(0,0,0,max_len-lang.shape[1])).float()
                    for B in range(lang.shape[0]):
                        for L in range(lang.shape[1]):
                            if lang[B][L].sum() == 0:
                                lang[B][L][0] = 1

                if cuda:
                    img = img.cuda()
                    y = y.cuda()
                    lang = lang.cuda()
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
                        langs, lang_lengths = speaker.sample(img, y)
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
                elif run_type == 'rsa':
                    langs = 0
                    lang_lengths = 0
                    for color in [4, 6, 9, 10, 11, 14, 0]:
                        for shape in [7, 8, 12, 13, 5]:
                            if color == 0:
                                lang = torch.zeros(batch_size, max_len, len(vocab['w2i'].keys())).to(lang.device)
                                lang[:, 0, data.SOS_IDX] = 1
                                lang[:, 1, shape] = 1
                                lang[:, 2, data.EOS_IDX] = 1
                                lang[:, 3:, data.PAD_IDX] = 1
                                lang_length = 3*torch.ones(batch_size)
                            else:
                                lang = torch.zeros(batch_size, max_len, len(vocab['w2i'].keys())).to(lang.device)
                                lang[:, 0, data.SOS_IDX] = 1
                                lang[:, 1, color] = 1
                                lang[:, 2, shape] = 1
                                lang[:, 3, data.EOS_IDX] = 1
                                lang[:, 4:, data.PAD_IDX] = 1
                                lang_length = 4*torch.ones(batch_size)
                            try:
                                langs = torch.cat((langs, lang.unsqueeze(0)), 0)
                                lang_lengths = torch.cat((lang_lengths, lang_length.unsqueeze(0)), 0)
                            except:
                                langs = lang.unsqueeze(0)
                                lang_lengths = lang_length.unsqueeze(0)
                elif run_type == 'test':
                    langs = torch.zeros(batch_size, max_len, len(vocab['w2i'].keys()))
                    for i in range(len(lang)):
                        color = lang[i,1]
                        shape = lang[i,2]
                        if test_type == 'color':
                            langs[i, 0, data.SOS_IDX] = 1
                            langs[i, 1, color] = 1
                            langs[i, 2, data.EOS_IDX] = 1
                            langs[i, 3:, data.PAD_IDX] = 1
                            lang_lengths = 3*torch.ones(batch_size)
                        if test_type == 'shape':
                            langs[i, 0, data.SOS_IDX] = 1
                            langs[i, 1, shape] = 1
                            langs[i, 2, data.EOS_IDX] = 1
                            langs[i, 3:, data.PAD_IDX] = 1
                            lang_lengths = 3*torch.ones(batch_size)
                        if test_type == 'color-shape':
                            langs[i, 0, data.SOS_IDX] = 1
                            langs[i, 1, color] = 1
                            langs[i, 2, shape] = 1
                            langs[i, 3, data.EOS_IDX] = 1
                            langs[i, 4:, data.PAD_IDX] = 1
                            lang_lengths = 4*torch.ones(batch_size)
                        if test_type == 'shape-color':
                            langs[i, 0, data.SOS_IDX] = 1
                            langs[i, 1, shape] = 1
                            langs[i, 2, color] = 1
                            langs[i, 3, data.EOS_IDX] = 1
                            langs[i, 4:, data.PAD_IDX] = 1
                            lang_lengths = 4*torch.ones(batch_size)
                    langs = langs.unsqueeze(0)
                    lang_lengths = lang_lengths.unsqueeze(0)
                elif run_type == 'pretrained':
                    if penalty == 'probability' or penalty == None:
                        lang, lang_length, eos_loss, lang_prob = speaker(img, y, activation = activation, tau = tau, length_penalty = False)
                    else:
                        lang, lang_length, eos_loss, lang_prob = speaker(img, y, activation = activation, tau = tau, length_penalty = True)
                elif run_type == 'oracle':
                    lang_length = length
                else:
                    lang, lang_length, eos_loss = speaker(img, y, activation = activation, tau = tau)

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
                    lang_out = lang_out.view(batch_size*lang_out.size(1), len(vocab['w2i'].keys()))
                    lang = lang.long().view(batch_size*lang.size(1), len(vocab['w2i'].keys()))
                    this_loss = loss(lang_out.cuda(), torch.max(lang, 1)[1])

                    if split == 'train':
                        # SGD step
                        this_loss.backward()
                        optimizer.step()
                        
                    this_acc = (lang_out.argmax(1)==lang.argmax(1)).float().mean().item()

                    meters['loss'].update(this_loss, batch_size)
                    meters['acc'].update(this_acc, batch_size)
                else:
                    if game_type != 'concept':
                        if run_type == 'sample' or run_type == 'rsa' or run_type == 'test':
                            if run_type == 'sample':
                                alpha = 20
                            else:
                                alpha = 0#10
                            if run_type != 'test':
                                best_score_diff = -math.inf*torch.ones(batch_size)
                                best_lang = torch.zeros((langs.shape[1],langs.shape[2],langs.shape[3]))
                                best_lang_length = torch.zeros(lang_lengths.shape[1])
                                for lang, lang_length in zip(langs, lang_lengths):
                                    lis_scores = internal_listener(img, lang, lang_length)
                                    for game in range(batch_size):
                                        score_diff = (lis_scores[game][y[game]]-np.delete(lis_scores[game].cpu(), y[game].cpu(), axis=0).mean()-alpha*lang_length[game]).cpu()
                                        if score_diff>best_score_diff[game]:
                                            best_score_diff[game] = score_diff
                                            best_lang[game] = lang[game]
                                            best_lang_length[game] = lang_length[game]

                                lang = best_lang
                                lang_length = best_lang_length
                                lis_scores = listener(img, lang, lang_length)
                            else:
                                lang = langs.squeeze()
                                lang_length = lang_lengths.squeeze()
                                lis_scores = listener(img, lang, lang_length)
                            
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
                                meters, outputs = collect_outputs(meters, outputs, vocab, img, y, lang, lang_length, lis_pred, lis_scores, this_loss, this_acc, batch_size, ci_listeners, language_model, loss2)
                            else:
                                meters['loss'].update(this_loss, batch_size)
                                meters['acc'].update(this_acc, batch_size)
                        else:
                            if split == 'train' and run_type == 'pretrained' and activation == 'multinomial':
                                # Reinforce
                                lis_scores = listener(img, lang, lang_length, average=False)
                            elif split == 'train' and run_type == 'pretrained' and activation != 'gumbel' and activation != None:
                                lis_scores = listener(img, lang, lang_length, average=True)
                            else:
                                lang_onehot = lang.argmax(2)
                                if activation != 'gumbel' and activation != None:
                                    lang = F.one_hot(lang_onehot, num_classes = len(vocab['w2i'].keys())).cuda().float()
                                lang_length = []
                                for seq in lang_onehot: lang_length.append(np.where(seq.cpu()==data.EOS_IDX)[0][0]+1)
                                lang_length = torch.tensor(lang_length).cuda()
                                lis_scores = listener(img, lang, lang_length)
                            # Evaluate loss and accuracy
                            if run_type == 'pretrain':
                                this_loss = loss(lis_scores, y.long())
                            elif run_type == 'pretrained':
                                
                                if penalty == 'both':
                                    lm_seq = language_model(lang,lang_length)[:, :-1].contiguous()
                                    temp_prob = loss2(lm_seq.view(batch_size*lm_seq.size(1),len(vocab['w2i'].keys())).cuda(),torch.max(lang.cuda()[:,1:].contiguous().long().view(batch_size*lm_seq.size(1),len(vocab['w2i'].keys())),1)[1])
                                    temp_prob = temp_prob.view(batch_size,lm_seq.size(1))
                                    for i,prob in enumerate(temp_prob):
                                        for j in range(lang_length[i].int(),max_len):
                                            temp_prob[:,j-1] = 0
                                        #seq_prob.append(math.exp(log_prob[-1]))
                                    log_prob = temp_prob.sum(1).mean()
                                    eos_loss = eos_loss+log_prob
                                    #print(eos_loss)
                                    """
                                    max_len = 40
                                    true_lang_length = torch.tensor([np.count_nonzero(t) for t in true_lang.cpu()], dtype=np.int)
                                    true_lang[true_lang>=len(vocab['w2i'].keys())] = 3
                                    true_lang = F.one_hot(true_lang, num_classes = len(vocab['w2i'].keys()))
                                    true_lang = F.pad(true_lang,(0,0,0,max_len-true_lang.shape[1])).float()
                                    for B in range(true_lang.shape[0]):
                                        for L in range(true_lang.shape[1]):
                                            if true_lang[B][L].sum() == 0:
                                                true_lang[B][L][0] = 1
                                    true_lis_scores = listener(img, true_lang, true_lang_length)
                                    """
                                if penalty == 'probability':
                                    
                                    lm_seq = language_model(lang,lang_length)[:, :-1].contiguous()
                                    temp_prob = loss2(lm_seq.view(batch_size*lm_seq.size(1),len(vocab['w2i'].keys())).cuda(),torch.max(lang.cuda()[:,1:].contiguous().long().view(batch_size*lm_seq.size(1),len(vocab['w2i'].keys())),1)[1])
                                    temp_prob = temp_prob.view(batch_size,lm_seq.size(1))
                                    for i,prob in enumerate(temp_prob):
                                        for j in range(lang_length[i].int(),max_len):
                                            temp_prob[:,j-1] = 0
                                        #seq_prob.append(math.exp(log_prob[-1]))
                                    log_prob = temp_prob.sum(1).mean()
                                    eos_loss = log_prob
                                    #print(eos_loss)
                                    """
                                    max_len = 40
                                    true_lang_length = torch.tensor([np.count_nonzero(t) for t in true_lang.cpu()], dtype=np.int)
                                    true_lang[true_lang>=len(vocab['w2i'].keys())] = 3
                                    true_lang = F.one_hot(true_lang, num_classes = len(vocab['w2i'].keys()))
                                    true_lang = F.pad(true_lang,(0,0,0,max_len-true_lang.shape[1])).float()
                                    for B in range(true_lang.shape[0]):
                                        for L in range(true_lang.shape[1]):
                                            if true_lang[B][L].sum() == 0:
                                                true_lang[B][L][0] = 1
                                    true_lis_scores = listener(img, true_lang, true_lang_length)
                                    """
                                if activation == 'multinomial':
                                    # Compute policy loss
                                    returns = (lis_scores.argmax(1) == y)
                                    # No reward for saying nothing
                                    not_zero = lang_length > 2
                                    returns = (returns & not_zero).float()
                                    # Slight negative reward for getting things wrong
                                    # (TODO: tweak this)
                                    returns = (1 * returns) + (-0.1 * (1 - returns))
                                    # FIXME: Should we normalize in the binary case?

                                    policy_loss = (-lang_prob * returns).sum()
                                    this_loss = policy_loss
                                else:
                                    this_loss = loss(lis_scores,y.long())
                                this_loss = this_loss + eos_loss * float(lm_wt)
                                """
                                print_it = True
                                for param in speaker.parameters():
                                    if print_it == True:
                                        print(param)
                                        print_it = False
                                """
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
                                meters, outputs = collect_outputs(meters, outputs, vocab, img, y, lang, lang_length, lis_pred, lis_scores, this_loss, this_acc, batch_size, ci_listeners, language_model, loss2)
                            else:
                                meters['loss'].update(this_loss-eos_loss*float(lm_wt), batch_size) #-prob*0.5, batch_size) #-seq_prob.detach()*0.005, batch_size) #-eos_prob*0.01, batch_size)
                                meters['lm loss'].update(eos_loss*float(lm_wt), batch_size)
                                meters['acc'].update(this_acc, batch_size)
                                meters['length'].update(lang_length.float().mean(), batch_size)
                                
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
        meters['length'] = np.array(meters['length']).tolist()
        meters['colors'] = np.array(meters['colors']).tolist()
        meters['shapes'] = np.array(meters['shapes']).tolist()
        metrics = meters
    else:
        metrics = compute_average_metrics(meters)
    seq = []
    for word_index in lang.argmax(2)[0,:].cpu().numpy():
        try:
            seq.append(vocab['i2w'][word_index])
        except:
            seq.append('<UNK>')
    print(seq)
    seq = []
    for word_index in true_lang[0,:].cpu().numpy():
        try:
            seq.append(vocab['i2w'][word_index])
        except:
            seq.append('<UNK>')
    print(seq)
    return metrics, outputs