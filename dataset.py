import numpy as np
import torch
import pickle as pkl
# todo add more imports to other files here... (datasets/training scripts/etc)
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate


# utils ------------------------------------
def load_pkl_file(filename):
    with open(filename, "rb") as f:
        x = pkl.load(f)
    return x

def invert_dict(d):
    return {v: k for k, v in d.items()}

def tokenize(s, delim=' ',
      add_start_token=True, add_end_token=True,
      punct_to_keep=None, punct_to_remove=None):
    """
    Tokenize a sequence, converting a string s into a list of (string) tokens by
    splitting on the specified delimiter. Optionally keep or remove certain
    punctuation marks and add start and end tokens.
    """
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, '%s%s' % (delim, p))
    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, '')
            
    tokens = s.split(delim)
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')
    return tokens

def encode(seq_tokens, token_to_idx, allow_unk=False):
    seq_idx = []
    for token in seq_tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = '<UNK>'
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        seq_idx.append(token_to_idx[token])
    return seq_idx

def decode(seq_idx, idx_to_token, delim=None, stop_at_end=True):
    tokens = []
    for idx in seq_idx:
        tokens.append(idx_to_token[idx])
        if stop_at_end and tokens[-1] == '<END>':
            break
    if delim is None:
        return tokens
    else:
        return delim.join(tokens)

SPECIAL_TOKENS = {
  '<NULL>': 0,
  '<START>': 1,
  '<END>': 2,
  '<UNK>': 3,
  '<DIA>': 4,
}
# end ------------------------------------


class CiCDataset(Dataset):
    """ Dataset wrapper for the CiC Dataset
    """
    SUPPORTED_VIS_FEATS = {
        "vgg" : "vgg_fc7_pt.pkl",
        # above is the official feature set, processed to be compatible with python3 pickle
        # todo: add other feature sets here from NES exps...
    }
    VOCAB_DEFAULT_FILENAME = "vocab_all.pkl"  # for now, assuming a single consistent vocab default file.
    
    def __init__(self,
                 dataset_prefix="./data/shapeglot/",
                 task="language",  # for main language generalization task 
                 vocab_tkn_to_idx=None,  # use this to fix the token_to_idx mapping; "None" will load the default vocab file.
                 vis_feats="vgg",
                 dataset_split="train",
                 **kwargs):
        super().__init__()
        self.dataset_prefix = dataset_prefix
        self.task = task
        self.dataset_split = dataset_split
        self.kwargs = kwargs # store to enable synchronizing between alternate splits...
        assert self.dataset_split in ["train", "val", "test"], f"unsupported split {dataset_split}"
        verbose_init = kwargs.get("verbose", True)
        # load the visual_features; logic here allows sharing a global visual feats dictionary across train-val split
        if isinstance(vis_feats, str):
            if vis_feats not in CiCDataset.SUPPORTED_VIS_FEATS:
                raise ValueError(f"unrecognized vis_feats: '{vis_feats}'")
            vis_feats_fname = f"{dataset_prefix}/{CiCDataset.SUPPORTED_VIS_FEATS[vis_feats]}"
            self.vis_feats = load_pkl_file(vis_feats_fname)
            print(f"loaded {vis_feats_fname}")
        else:
            print(f"using vis_feats dictionary that was passed in.")
            self.vis_feats = vis_feats
        print(f"total number of vis_feats = {len(self.vis_feats.keys())}")
        
        
        # load the vocab
        self.token_to_idx = vocab_tkn_to_idx
        if vocab_tkn_to_idx is None:
            vocab_fname = f"{dataset_prefix}/{task}/{CiCDataset.VOCAB_DEFAULT_FILENAME}"
            print(f"loading vocab file from: {vocab_fname}")
            self.token_to_idx = load_pkl_file(vocab_fname)
        else:
            print(f"using provided vocab_tkn_to_idx mapping!")
        print(f"total vocab size = {len(self.token_to_idx.keys())}")
        self.idx_to_token = invert_dict(self.token_to_idx)
        
        # load the split-specific language and metadata
        split_fname = f"{dataset_prefix}/{task}/{dataset_split}.pkl"
        print(f"loading dataset split file: {split_fname}")
        split_data = load_pkl_file(split_fname)
        self.game_id = split_data["game_id"]
        self.trial_num = split_data["trial_num"]
        self.context_condition = split_data["context_condition"]  # not really used for anything right now...
        
        self.chair_a = split_data["chair_a"]
        self.chair_b = split_data["chair_b"]
        self.chair_c = split_data["chair_c"]
        self.target_chair = split_data["target_chair"]
        
        self.text = split_data["glove_corrected_text"]
        
        self.size = len(self.game_id)  # todo add check to ensure other fields have same size.
        print(f"dataset split {self.dataset_split} info loaded with {self.size} entries.")
        
        self.minimal_batches = kwargs.get("minimal_batches", False)
        print(f"self.minimal_batches flag is set to {self.minimal_batches}")
        if self.minimal_batches:
            raise NotImplementedError("todo: minimal_batches collate function support")
        ## end init ##
        
        
    def __len__(self):
        return self.size
            
    def get_vocab(self, direction=None):
        d_vocab = {
            "token_to_idx" : self.token_to_idx,
            "idx_to_token" : self.idx_to_token
        }
        if direction:
            return d_vocab[direction]
        return d_vocab        
            
    def __getitem__(self, index):
        '''
        dataset is effectively iterated over as:
          (world0, caption0)
          (world0, caption1)
              ... (rest of captions) ...
          (world0, captionN)
          (world1, caption0)
              ... etc.
        '''
        # get metadata
        game_id = self.game_id[index]
        trial_num = self.trial_num[index]
        context_condition = self.context_condition[index]
        
        # get chair keys
        chair_key_a = self.chair_a[index]
        chair_key_b = self.chair_b[index]
        chair_key_c = self.chair_c[index]
        
        # get visual data
        vis_chair_a = torch.tensor(self.vis_feats[chair_key_a]).contiguous().float()
        vis_chair_b = torch.tensor(self.vis_feats[chair_key_b]).contiguous().float()
        vis_chair_c = torch.tensor(self.vis_feats[chair_key_c]).contiguous().float()
        
        # get language
        text = self.text[index]
        q_tokens = tokenize(text, add_start_token=True, add_end_token=True)
        q = torch.LongTensor(encode(q_tokens, self.token_to_idx, allow_unk=True))
        q_len = int(len(q_tokens))

        # get target chair label
        target_chair = int(self.target_chair[index])
        
        if self.minimal_batches:
            return (game_id, trial_num,
                    vis_chair_a, vis_chair_b, vis_chair_c,
                    q_tokens, q, q_len, target_chair, index)
        
        return (game_id, trial_num, context_condition,
                chair_key_a, chair_key_b, chair_key_c,
                vis_chair_a, vis_chair_b, vis_chair_c,
                text, q_tokens, q, q_len,
                target_chair, index)
        
    def get_name(self, capitalize=True):
        dset_name = self.dataset_split
        return dset_name if not capitalize else dset_name.capitalize()

    def cic_collate(batch):
        # Collate Fn for DataLoader, corresponding to ShapeWorldDataset class
        # Enables full control by train script
        return batch
    
    def cic_collate_process_srt(batch):
        # Collate Fn for DataLoader, corresponding to ShapeWorldDataset class
        # Processes all the individual attributes.
        # This should be the standard collate fn passed in during training.
        grouped_batch = list(zip(*batch)) # group by entry type
        
        # metadata
        game_id, trial_num, context_condition = grouped_batch[0:3]
        # chair data
        chair_key_a, chair_key_b, chair_key_c = grouped_batch[3:6]
        vis_chair_a, vis_chair_b, vis_chair_c = grouped_batch[6:9]
        # language_data
        text, str_seq, v_seq, v_len = grouped_batch[9:13]
        # final data
        target_chair, index = grouped_batch[13:15]
        assert len(grouped_batch) == 15
        
        # sort by length of sequence to support pytorch sequence packing/unpacking operations.
        v_lens = default_collate(v_len)
        v_lens, perm_idx = v_lens.sort(0, descending=True)
        
        vis_chairs_a = default_collate(vis_chair_a)[perm_idx]
        vis_chairs_b = default_collate(vis_chair_b)[perm_idx]
        vis_chairs_c = default_collate(vis_chair_c)[perm_idx]
        
        targets = default_collate(target_chair)[perm_idx]
        indices = default_collate(index)[perm_idx]
        
        str_seqs, v_seqs = zip(*[(str_seq[i], v_seq[i]) for i in perm_idx])
        
        # misc stuff        
        perm_idx_np = perm_idx.clone().numpy()
        for i_np, i_pt in zip(perm_idx_np, perm_idx): # todo remove this...
            assert int(i_np) == int(i_pt)
        metadata = {
            "game_id" : np.array(game_id)[perm_idx_np],
            "trial_num" : np.array(trial_num)[perm_idx_np],
            "context_condition" : np.array(context_condition)[perm_idx_np],
            "text" : np.array(text)[perm_idx_np],
            "chair_key_a" : np.array(chair_key_a)[perm_idx_np],
            "chair_key_b" : np.array(chair_key_b)[perm_idx_np],
            "chair_key_c" : np.array(chair_key_c)[perm_idx_np],
            "indices" : indices,
            "perm_idx_np" : perm_idx_np,
        }
        
        vis_chairs = (vis_chairs_a, vis_chairs_b, vis_chairs_c)
        return vis_chairs, str_seqs, v_seqs, v_lens, targets, metadata



def get_other_split(dset, new_split):
    return CiCDataset(
                dataset_prefix=dset.dataset_prefix,
                task=dset.task,
                vocab_tkn_to_idx=dset.token_to_idx, # this will be a dict now
                vis_feats=dset.vis_feats, # this will be a dict now
                dataset_split=new_split, **dset.kwargs)

