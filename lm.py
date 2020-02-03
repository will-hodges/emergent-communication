import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
import torch.utils.data as du
import numpy as np
from tqdm import tqdm
from util import AverageMeter
from glob import glob
from collections import defaultdict
import contextlib


class LM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(len(vocab['w2i']), 300)
        self.rnn = nn.GRU(300, 512, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(512, len(vocab['w2i']))
        self.vocab = vocab
        self.criterion = nn.CrossEntropyLoss()

    def set_forward(self, lang, lang_length):
        embs = self.emb(lang)

        inputs = rnn_utils.pack_padded_sequence(embs, lang_length - 1, batch_first=True, enforce_sorted=False)
        targets = rnn_utils.pack_padded_sequence(lang[:, 1:], lang_length - 1, batch_first=True, enforce_sorted=False)
        assert (inputs.sorted_indices == targets.sorted_indices).all()
        targets = targets.data

        hidden, _ = self.rnn(inputs)

        preds = self.linear(self.dropout(hidden.data))

        return preds, targets


    def forward(self, lang, lang_length):
        preds, targets = self.set_forward(lang, lang_length)
        loss = self.criterion(preds, targets)

        return loss


lang = []
for fname in glob('/mnt/fs2/juliawhi/emergent/data/colors/data_1000*'):
    lang.append(np.load(fname)['langs'])
lang = np.concatenate(lang, 0)
lang_length = (lang != 0).sum(1)

MAX_LANG_LENGTH = lang_length.max()

vocab = torch.load('/mnt/fs2/juliawhi/emergent/models/colors/vocab.pt')

def load_text(fname):
    with open(fname, 'r') as f:
        lines = list(f)
    all_tokens = []
    for line in lines:
        if line.startswith('['):
            line = eval(line)
        else:
            line = line.strip().split()
        tokens = []
        for t in line:
            if t == '<PAD>':
                break
            i = vocab['w2i'].get(t, vocab['w2i']['<UNK>'])
            if t not in vocab['w2i']:
                print(f'oov: {t}')
            tokens.append(i)
        if len(tokens) == 1:
            continue
        all_tokens.append(tokens)
    len_t = torch.tensor([len(t) for t in all_tokens], dtype=torch.long)
    max_lang_len = len_t.max()
    lang_t = torch.zeros((len(all_tokens), max_lang_len), dtype=torch.long)
    for i, tks in enumerate(all_tokens):
        lang_t[i, :len_t[i]] = torch.tensor(tks)
    dset = du.TensorDataset(lang_t, len_t)
    dloader = du.DataLoader(dset, batch_size=32, shuffle=True)
    return dloader


eval_data = {
    's_context': load_text('text/s_context.txt'),
    's_amortized': load_text('text/s_amortized.txt'),
    's_srr5': load_text('text/srr5.txt'),
}


lang = torch.from_numpy(lang)
lang_length = torch.from_numpy(lang_length)


train_dset = du.TensorDataset(lang, lang_length)
train_loader = du.DataLoader(train_dset, batch_size=32, shuffle=True)

DEVICE = torch.device('cuda')

# Unkify out of vocab tokens
MAX_I = max(list(vocab['w2i'].values()))
lang[lang > MAX_I] = vocab['w2i']['<UNK>']

model = LM(vocab)
model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.Adam(model.parameters())

metrics = defaultdict(list)
metrics['best_loss'] = np.inf
metrics['best_epoch'] = 0
metrics['best_perplexity'] = np.inf

model.train()

def update(pbar, split, epoch, meters):
    pbar.set_description(f"Epoch {epoch} {split.upper()}: Loss {meters['loss'].avg:.3f} PPL {meters['ppl'].avg:.3f}")

def run(split, epoch, loader):
    training = split == 'train'
    if training:
        model.train()
        ctx = contextlib.nullcontext
    else:
        model.eval()
        ctx = torch.no_grad

    meters = {m: AverageMeter() for m in ['loss', 'ppl']}

    ranger = tqdm(loader)
    with ctx():
        for batch_i, (lang, lang_length) in enumerate(ranger):
            lang = lang[:, :lang_length.max()]
            lang = lang.to(DEVICE)
            lang_length = lang_length.to(DEVICE)

            if training:
                optimizer.zero_grad()

            loss = model(lang, lang_length)

            meters['loss'].update(loss.item(), lang.shape[0])
            meters['ppl'].update(loss.exp().item(), lang.shape[0])

            if training:
                loss.backward()
                optimizer.step()
            if batch_i % 100 == 0:
                update(ranger, split, epoch, meters)

    update(ranger, split, epoch, meters)
    return {k: m.avg for k, m in meters.items()}

for epoch in range(20):
    print(f'==== EPOCH {epoch} ====')
    metrics['epoch'] = epoch

    train_metrics = run('train', epoch, train_loader)
    for name, metric in train_metrics.items():
        metrics[f'train_{name}'].append(metric)

    val_metrics = run('val', epoch, train_loader)
    for name, metric in val_metrics.items():
        metrics[f'val_{name}'].append(metric)

    if val_metrics['loss'] < metrics['best_loss']:
        metrics['best_loss'] = val_metrics['loss']
        metrics['best_perplexity'] = val_metrics['ppl']
        metrics['best_epoch'] = epoch
        torch.save(model.state_dict(), './lm.pth')

    # Eval
    for name, name_loader in eval_data.items():
        name_metrics = run(name, epoch, name_loader)
