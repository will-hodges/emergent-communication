import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import data
from data import ShapeWorld
import statistics
import matplotlib.pyplot as plt
from glob import glob
import os

def evaluate_listener(l0, data_file, vocab, batch_size, cuda):
    context = torch.no_grad()
    historical_acc = []
    d = data.load_raw_data(data_file)
    dataloader = DataLoader(ShapeWorld(d, vocab), batch_size=batch_size, shuffle=False)
    
    with context:
        for batch_i, (img, y, lang) in enumerate(dataloader):
            batch_size = img.shape[0]

            # Reformat inputs
            y = y.argmax(1)
            img = img.float()

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
                length = length.cuda()


            
            # Give language to listener and get prediction
            lis_scores = l0(img, lang, length)
            lis_pred = F.softmax(lis_scores).argmax(1)            
            correct = [a == b for a, b in zip(lis_pred.tolist(), y.tolist())]
            acc = correct.count(True) / len(correct)
            historical_acc.append(acc)
        return statistics.mean(historical_acc)

def evaluate_speaker(sl0, l0, data_file, vocab, batch_size, cuda, dataset, debug=False):
    context = torch.no_grad()
    d = data.load_raw_data(data_file)
    dataloader = DataLoader(ShapeWorld(d, vocab), batch_size=batch_size, shuffle=False)
    
    historical_acc = []
    
    with context:
        for batch_i, (img, y, lang) in enumerate(dataloader):
            batch_size = img.shape[0]

            # Reformat inputs
            y = y.argmax(1)
            img = img.float()

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
                length = length.cuda()

            # Get speaker language
            lang_out = sl0(img, lang, length, y)
            if dataset != 'shapeglot':
                pred_text = dataloader.dataset.to_text(lang_out.argmax(2))[0] # Human readable
                actual_text = dataloader.dataset.to_text(lang.argmax(2))[0]
            else:
                pred_text = dataloader.dataset.gettext(lang_out.argmax(2))[0]
                actual_text = dataloader.dataset.gettext(lang.argmax(2))[0]
            if debug:
                print(f'Text from speaker: {pred_text}')
                print(f'Actual text: {actual_text}')
            
            # Give language to listener and get prediction
            lis_scores = l0(img, lang_out, length)
            lis_pred = F.softmax(lis_scores).argmax(1)
            
            correct = [a == b for a, b in zip(lis_pred.tolist(), y.tolist())]
            acc = correct.count(True) / len(correct)
            historical_acc.append(acc)
        return statistics.mean(historical_acc)
    
if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    
    parser = ArgumentParser(description='Evaluate a literal speaker', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sl0', default='models/shapeglot/actual_literal_speaker.pt', help='path to literal speaker')
    parser.add_argument('--s0', default='models/shapeglot/literal_speaker.pt', help='path to contextual speaker')
    parser.add_argument('--l0', default='models/shapeglot/pretrained_listener_0.pt', help='path to literal listener')
    parser.add_argument('--dataset', default='shapeglot', help='chairs, colors, shapeglot, or shapeworld')
    parser.add_argument('--plot_title', default='Accuracy of Models', help='title for plot')
    parser.add_argument('--save', default='accuracy.png', help='path to savefile')
    parser.add_argument('--split', default='val', help='which split to evaluate on')
    parser.add_argument('--cuda', action='store_true', help='run with cuda')
    parser.add_argument('--debug', action='store_true', help='print output')
    parser.add_argument('--batch_size', default=32, type=int)
    args = parser.parse_args()
    
    if args.split not in ['val', 'train', 'test']:
        raise Exception('Split not recognized. Must be val, train, or test')
    
    # Convert paths to models
    s0 = torch.load(args.s0)
    sl0 = torch.load(args.sl0)
    l0 = torch.load(args.l0)
    
    # Load data to test on
    if args.dataset == 'chairs':
        data_dir = './data/chairs/data_1000_'
    elif args.dataset == 'colors':
        data_dir = './data/colors/data_1000_'
    elif args.dataset == 'shapeworld':
        data_dir = './data/single/reference-1000-'
    elif args.dataset == 'shapeglot':
        data_dir = './data/shapeglot/data_1000_'
    else:
        raise Exception('Dataset ' + args.dataset + ' is not defined.')
        
    # Load .npz files and the vocab
    if args.dataset != 'shapeglot':
        data_files = [data_dir + str(e) + '.npz' for e in range(15,30)]
    else:
        data_files = glob(os.path.join(f'data/shapeglot/*_{args.split}_*.npz'))
    vocab = torch.load('./models/' + args.dataset + '/vocab.pt')

    
    if args.cuda:
        s0.cuda()
        sl0.cuda()
        l0.cuda()
    s0.eval()
    sl0.eval()
    l0.eval()
    
    y = []
    z = []
    k = []
    
    x = []
    
    epoch = 0
    for file in data_files:
        y.append(evaluate_speaker(sl0, l0, file, vocab, args.batch_size, args.cuda, args.dataset, args.debug))
        x.append(epoch)
        epoch += 1
        
    for file in data_files:
        z.append(evaluate_speaker(s0, l0, file, vocab, args.batch_size, args.cuda, args.dataset, args.debug))
                                 
    for file in data_files:
        k.append(evaluate_listener(l0, file, vocab, args.batch_size, args.cuda))
        
        
    a = plt.subplot(111)
    literal_speaker = a.bar([b-0.2 for b in x], y, width=0.2, color='b', align='center', label='S0')
    contextual_speaker = a.bar(x, z, width=0.2, color='g', align='center', label='S\'0')
    literal_listener = a.bar([b+0.2 for b in x], k, width=0.2, color='r', align='center', label='L0')
    
    a.legend((literal_speaker[0], contextual_speaker[0], literal_listener[0]),('S0','S\'0','L0')) 
    plt.title(args.plot_title)
    plt.xlabel("split")
    plt.ylabel("accuracy")
    plt.savefig(args.save)
    