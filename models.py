"""
RNN encoders and decoders
"""

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.utils.rnn as rnn_utils
import data
from train import VOCAB


class FeatureMLP(nn.Module):
    def __init__(self, input_size=16, output_size=16):
        super(FeatureMLP, self).__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size)
        )
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        return self.trunk(x)


def to_onehot(y, n=3):
    y_onehot = torch.zeros(y.shape[0], n).to(y.device)
    y_onehot.scatter_(1, y.view(-1, 1), 1)
    return y_onehot


class Speaker(nn.Module):
    def __init__(self, feat_model, embedding_module, game, hidden_size=100):
        super(Speaker, self).__init__()
        self.embedding = embedding_module
        self.feat_model = feat_model
        self.feat_size = feat_model.final_feat_dim
        self.embedding_dim = embedding_module.embedding_dim
        self.vocab_size = embedding_module.num_embeddings
        self.game = game
        self.hidden_size = hidden_size

        self.gru = nn.GRU(self.embedding_dim, self.hidden_size)
        self.outputs2vocab = nn.Linear(self.hidden_size, self.vocab_size)
        # n_obj of feature size + 1/0 indicating target index
        if game == 'reference':
            self.init_h = nn.Linear(3 * (self.feat_size + 1), self.hidden_size)
        if game == 'concept':
            self.init_h = nn.Linear(10 * (self.feat_size + 1), self.hidden_size)

    def embed_features(self, feats, targets):
        batch_size = feats.shape[0]
        n_obj = feats.shape[1]
        rest = feats.shape[2:]
        feats_flat = feats.view(batch_size * n_obj, *rest)
        feats_emb_flat = self.feat_model(feats_flat)

        feats_emb = feats_emb_flat.unsqueeze(1).view(batch_size, n_obj, -1)

        # Add targets
        if self.game == 'reference':
            targets_onehot = to_onehot(targets)
        if self.game == 'concept':
            targets_onehot = targets.float()

        feats_and_targets = torch.cat((feats_emb, targets_onehot.unsqueeze(2)), 2)

        ft_concat = feats_and_targets.view(batch_size, -1)

        return ft_concat

    def forward(self, feats, targets, greedy=False, max_len=20):
        """Sample from image features"""
        batch_size = feats.size(0)

        feats_emb = self.embed_features(feats, targets)

        # initialize hidden states using image features
        states = self.init_h(feats_emb)
        states = states.unsqueeze(0)

        # This contains are series of sampled onehot vectors
        lang = []
        # And vector lengths
        lang_length = torch.ones(batch_size, dtype=torch.int64).to(feats.device)
        done_sampling = [False for _ in range(batch_size)]

        # first input is SOS token
        # (batch_size, n_vocab)
        inputs_onehot = torch.zeros(batch_size, self.vocab_size).to(feats.device)
        inputs_onehot[:, data.SOS_IDX] = 1.0

        # (batch_size, len, n_vocab)
        inputs_onehot = inputs_onehot.unsqueeze(1)

        # Add SOS to lang
        lang.append(inputs_onehot)

        # (B,L,D) to (L,B,D)
        inputs_onehot = inputs_onehot.transpose(0, 1)

        # compute embeddings
        # (1, batch_size, n_vocab) X (n_vocab, h) -> (1, batch_size, h)
        inputs = inputs_onehot @ self.embedding.weight

        for i in range(max_len - 2):  # Have room for SOS, EOS if never sampled
            # FIXME: This is inefficient since I do sampling even if we've
            # finished generating language.
            if all(done_sampling):
                break
            outputs, states = self.gru(inputs, states)  # outputs: (L=1,B,H)
            outputs = outputs.squeeze(0)                # outputs: (B,H)
            outputs = self.outputs2vocab(outputs)       # outputs: (B,V)
            
            if greedy:
                predicted = outputs.max(1)[1]
                predicted = predicted.unsqueeze(1)
            else:
                #  outputs = F.softmax(outputs, dim=1)
                #  predicted = torch.multinomial(outputs, 1)
                # TODO: Need to let language model accept one-hot vectors.
                predicted_onehot = F.gumbel_softmax(outputs, tau=1, hard=True)
                # Add to lang
                lang.append(predicted_onehot.unsqueeze(1))

            predicted_npy = predicted_onehot.argmax(1).cpu().numpy()
            
            # Update language lengths
            for i, pred in enumerate(predicted_npy):
                if not done_sampling[i]:
                    lang_length[i] += 1
                if pred == data.EOS_IDX:
                    done_sampling[i] = True

            # (1, batch_size, n_vocab) X (n_vocab, h) -> (1, batch_size, h)
            inputs = (predicted_onehot.unsqueeze(0)) @ self.embedding.weight

        # Add EOS if we've never sampled it
        eos_onehot = torch.zeros(batch_size, 1, self.vocab_size).to(feats.device)
        eos_onehot[:, 0, data.EOS_IDX] = 1.0
        lang.append(eos_onehot)
        # Cut off the rest of the sentences
        for i, _ in enumerate(predicted_npy):
            if not done_sampling[i]:
                lang_length[i] += 1
            done_sampling[i] = True

        # Cat language tensors
        lang_tensor = torch.cat(lang, 1)

        # Trim max length
        max_lang_len = lang_length.max()
        lang_tensor = lang_tensor[:, :max_lang_len, :]

        return lang_tensor, lang_length

    def to_text(self, lang_onehot):
        texts = []
        lang = lang_onehot.argmax(2)
        for sample in lang.cpu().numpy():
            text = []
            for item in sample:
                text.append(data.ITOS[item])
                if item == data.EOS_IDX:
                    break
            texts.append(' '.join(text))
        return np.array(texts, dtype=np.unicode_)


class LiteralSpeaker(nn.Module):
    def __init__(self, feat_model, embedding_module, hidden_size=100):
        """super(Speaker, self).__init__()
        self.embedding = embedding_module
        self.feat_model = feat_model
        self.feat_size = feat_model.final_feat_dim
        self.embedding_dim = embedding_module.embedding_dim
        self.vocab_size = embedding_module.num_embeddings
        self.game = game
        self.hidden_size = hidden_size"""
        
        super(LiteralSpeaker, self).__init__()
        self.embedding = embedding_module
        self.embedding_dim = embedding_module.embedding_dim
        self.vocab_size = embedding_module.num_embeddings
        self.hidden_size = hidden_size
        self.feat_model = feat_model
        self.feat_size = feat_model.final_feat_dim
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size)
        self.outputs2vocab = nn.Linear(self.hidden_size, self.vocab_size)
        self.init_h = nn.Linear(self.feat_size, self.hidden_size)

    def forward(self, feats, seq, length, y):
            
        feats = torch.from_numpy(np.array([np.array(feat[y[idx],:,:,:].cpu()) for idx, feat in enumerate(feats)]))
            
        # feats is from example images
        batch_size = seq.shape[0]
        if batch_size > 1:
            # BUGFIX? dont we need to sort feats too?
            sorted_lengths, sorted_idx = torch.sort(length, descending=True)
            seq = seq[sorted_idx]
            feats = feats[sorted_idx]

        feats = feats.unsqueeze(0)
        # reorder from (B,L,D) to (L,B,D)
        seq = seq.transpose(0, 1)

        # embed your sequences
        embed_seq = seq @ self.embedding.weight
        
        # embed features
        feats_emb = self.feat_model(feats.squeeze().cuda())
        feats_emb = self.init_h(feats_emb)
        feats_emb = feats_emb.unsqueeze(0)
        
        # shape = (seq_len, batch, hidden_dim)
        output, _ = self.gru(embed_seq, feats_emb)

        # reorder from (L,B,D) to (B,L,D)
        output = output.transpose(0, 1)

        if batch_size > 1:
            _, reversed_idx = torch.sort(sorted_idx)
            output = output[reversed_idx]

        max_length = output.size(1)
        output_2d = output.view(batch_size * max_length, -1)
        outputs_2d = self.outputs2vocab(output_2d)
        lang_tensor = outputs_2d.view(batch_size, max_length, self.vocab_size)
        
        return lang_tensor
    
    def sample(self, feats, sos_index, eos_index, pad_index, greedy=False):
        """Generate from image features using greedy search."""
        with torch.no_grad():
            batch_size = feats.size(0)

            # initialize hidden states using image features
            states = feats.unsqueeze(0)

            # first input is SOS token
            inputs = np.array([sos_index for _ in range(batch_size)])
            inputs = torch.from_numpy(inputs)
            inputs = inputs.unsqueeze(1)
            inputs = inputs.to(feats.device)

            # save SOS as first generated token
            inputs_npy = inputs.squeeze(1).cpu().numpy()
            sampled_ids = [[w] for w in inputs_npy]

            # (B,L,D) to (L,B,D)
            inputs = inputs.transpose(0, 1)

            # compute embeddings
            inputs = self.embedding(inputs)

            for i in range(20):  # like in jacobs repo
                outputs, states = self.gru(inputs, states)  # outputs: (L=1,B,H)
                outputs = outputs.squeeze(0)                # outputs: (B,H)
                outputs = self.outputs2vocab(outputs)       # outputs: (B,V)

                if greedy:
                    predicted = outputs.max(1)[1]
                    predicted = predicted.unsqueeze(1)
                else:
                    outputs = F.softmax(outputs, dim=1)
                    predicted = torch.multinomial(outputs, 1)

                predicted_npy = predicted.squeeze(1).cpu().numpy()
                predicted_lst = predicted_npy.tolist()

                for w, so_far in zip(predicted_lst, sampled_ids):
                    if so_far[-1] != eos_index:
                        so_far.append(w)

                inputs = predicted.transpose(0, 1)          # inputs: (L=1,B)
                inputs = self.embedding(inputs)             # inputs: (L=1,B,E)

            sampled_lengths = [len(text) for text in sampled_ids]
            sampled_lengths = np.array(sampled_lengths)

            max_length = max(sampled_lengths)
            padded_ids = np.ones((batch_size, max_length)) * pad_index

            for i in range(batch_size):
                padded_ids[i, :sampled_lengths[i]] = sampled_ids[i]

            sampled_lengths = torch.from_numpy(sampled_lengths).long()
            sampled_ids = torch.from_numpy(padded_ids).long()

        return sampled_ids, sampled_lengths
    
class RNNEncoder(nn.Module):
    """
    RNN Encoder - takes in onehot representations of tokens, rather than numeric
    """
    def __init__(self, embedding_module, hidden_size=100):
        super(RNNEncoder, self).__init__()
        self.embedding = embedding_module
        self.embedding_dim = embedding_module.embedding_dim
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.embedding_dim, hidden_size)

    def forward(self, seq, length):
        batch_size = seq.size(0)

        if batch_size > 1:
            sorted_lengths, sorted_idx = torch.sort(length, descending=True)
            seq = seq[sorted_idx]

        # reorder from (B,L,D) to (L,B,D)
        seq = seq.transpose(0, 1)

        # embed your sequences
        embed_seq = seq @ self.embedding.weight

        packed = rnn_utils.pack_padded_sequence(
            embed_seq,
            sorted_lengths.data.tolist() if batch_size > 1 else length.data.tolist())

        _, hidden = self.gru(packed)
        hidden = hidden[-1, ...]

        if batch_size > 1:
            _, reversed_idx = torch.sort(sorted_idx)
            hidden = hidden[reversed_idx]

        return hidden


class Listener(nn.Module):
    def __init__(self, feat_model, embedding_module):
        super(Listener, self).__init__()
        self.embedding = embedding_module
        self.lang_model = RNNEncoder(self.embedding)
        self.feat_model = feat_model
        self.feat_size = feat_model.final_feat_dim
        self.vocab_size = embedding_module.num_embeddings
        self.bilinear = nn.Linear(self.lang_model.hidden_size, self.feat_size, bias=False)

    def embed_features(self, feats):
        batch_size = feats.shape[0]
        n_obj = feats.shape[1]
        rest = feats.shape[2:]
        feats_flat = feats.reshape(batch_size * n_obj, *rest)
        feats_emb_flat = self.feat_model(feats_flat)
        feats_emb = feats_emb_flat.unsqueeze(1).view(batch_size, n_obj, -1)
        return feats_emb

    def forward(self, feats, lang, lang_length):
        # Embed features
        feats_emb = self.embed_features(feats)

        # Embed language
        lang_emb = self.lang_model(lang, lang_length)

        # Bilinear term: lang embedding space -> feature embedding space
        lang_bilinear = self.bilinear(lang_emb)

        # Compute dot products
        scores = torch.einsum('ijh,ih->ij', (feats_emb, lang_bilinear))

        return scores
