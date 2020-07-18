import os
import sys
import copy
import json
import torch
import pickle
import random
import logging
import numpy as np
from tqdm import tqdm
from itertools import chain
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from chairs import ChairsInContext
from colors import ColorsInContext
from data import ShapeWorld
import dataset as shapeglot
        
if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(description='Convert .CSV file to processed .NPZ files', formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--data', default='shapeglot/chairs_group_data.csv', help='Path to .CSV file')
    parser.add_argument('--dataset', default='shapeglot', help='what dataset')
    parser.add_argument('--vocab', default='models/shapeglot/vocab.pt', help='Path to vocab file')
    parser.add_argument('--split', default=1000, help='When to save files')
    parser.add_argument('--image_size', default=64)
    parser.add_argument('--context_condition', default='all')
    parser.add_argument('--split_mode', default='easy')
    args = parser.parse_args()
    
    vocab = torch.load(args.vocab)
    
    if args.dataset == 'chairs' or args.dataset == 'shapeglot':
        DatasetClass = ChairsInContext
    elif args.dataset == 'colors':
        DatasetClass = ColorsInContext
    else:
        raise Exception('Dataset '+args.dataset+' is not defined.')
    
    data = DatasetClass('./'+str(args.dataset), image_size = args.image_size, vocab = vocab, split = 'train', context_condition = args.context_condition, train_frac = 1, val_frac = 0, image_transform = None)
    
    all_imgs = None
    all_labels = None
    langs = None
    count = 0
    
    for i, (img, y, lang) in enumerate(data):
        label = [0, 0, 0]
        label[y] = 1
        if all_imgs is None:
            all_imgs = np.array([img.numpy()])
            all_labels = np.array([label])
            langs = np.array([lang.numpy()])
        else:
            all_imgs = np.append(all_imgs,np.array([img.numpy()]),0)
            all_labels = np.append(all_labels,np.array([label]),0)
            langs = np.append(langs,np.array([lang.numpy()]),0)
        seq = []
        if (i+1)%args.split == 0:
            print("Saving...")
            print(langs)
            data_dict = {'imgs': all_imgs, 'labels': all_labels, 'langs': langs}
            np.savez_compressed('./data/'+str(args.dataset)+'/data_1000_'+str(count)+'.npz', **data_dict)
            count += 1
            all_imgs = None
            all_labels = None
            langs = None