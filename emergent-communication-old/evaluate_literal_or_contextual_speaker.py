import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import data
from data import ShapeWorld
import statistics

def evaluate(sl0, l0, data_file, vocab, batch_size, cuda):
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
            pred_text = dataloader.dataset.to_text(lang_out.argmax(2))[0] # Human readable
            
            # Give language to listener and get prediction
            lis_scores = l0(img, lang_out, length)
            lis_pred = F.softmax(lis_scores).argmax(1)
            
            correct = [a == b for a, b in zip(lis_pred.tolist(), y.tolist())]
            acc = correct.count(True) / len(correct)
            historical_acc.append(acc)
            print("L0 accuracy receiving text from SL0, batch %s: %s" % (batch_i, acc))
        print("L0 mean accuracy receiving text from SL0: %s" % statistics.mean(historical_acc))
    
if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    
    parser = ArgumentParser(description='Evaluate a literal speaker', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sl0', default='testing_models/actual_literal_speaker.pt', help='path to literal speaker')
    parser.add_argument('--l0', default='testing_models/pretrained_listener_0.pt', help='path to literal listener')
    parser.add_argument('--dataset', default='chairs', help='chairs, colors, shapeglot, or shapeworld')
    parser.add_argument('--cuda', action='store_true', help='run with cuda')
    parser.add_argument('--batch_size', default=32, type=int)
    args = parser.parse_args()
    
    # Convert paths to models
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
    data_files = [data_dir + str(e) + '.npz' for e in range(15,30)]
    vocab = torch.load('./models/' + args.dataset + '/vocab.pt')

    
    if args.cuda:
        sl0.cuda()
        l0.cuda()
    sl0.eval()
    l0.eval()
    
    for file in data_files:
        evaluate(sl0, l0, file, vocab, args.batch_size, args.cuda)
    