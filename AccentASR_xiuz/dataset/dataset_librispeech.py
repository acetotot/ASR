import os
import pickle
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from utils.feature_utils import zero_padding, target_padding
import pandas as pd
import yaml


# TODO : Move this to config
HALF_BATCHSIZE_TIME=800
HALF_BATCHSIZE_LABEL=150


class LibriDataset(Dataset):
    def __init__(self, root_path, sets, bucket_size, min_timestep=0, max_timestep=0, max_label_len=0, drop=False, text_only=False):
        """
        Init Librispeech (all datasets work in bucketing style)
        :param root_path:  str, file path to dataset
        :param sets: list of set
        :param bucket_size:  int, batch size for each bucket
        :param min_timestep:  int, min len for input (set to 0 for no restriction)
        :param max_timestep:  int, max len for input (set to 0 for no restriction)
        :param max_label_len:  int, max len for output (set to 0 for no restriction)
        :param drop: whether to drop long seq
        :param text_only: If true, only use y for LM
        """
        # Read file
        self.root = root_path  #存放CVS文件的路径
        tables = []
        for s in sets:  #sets分验证集dev和训练集train
            tables.append(pd.read_csv(F"dataset/{s}.csv")) #tables等于一个dev.cvs和train.cvs
        self.table = pd.concat(tables, ignore_index=True).sort_values(by=['LEN'], ascending=False)
        self.text_only = text_only

        # Crop seqs that are too long
        if drop and max_timestep > 0 and not text_only:
            self.table = self.table[self.table["LEN"] < max_timestep]
        # Crop seqs that are too long
        if drop and min_timestep > 0 and not text_only:
            self.table = self.table[self.table["LEN"] > min_timestep]
        if drop and max_label_len > 0:
            self.table = self.table[self.table["LABEL"].apply(lambda x: x.count(' '))+1 < max_label_len]

        X = self.table["PATH"].tolist()
        X_lens = self.table['LEN'].tolist()
        # Cast '0 1 2' to [0, 1, 2]
        Y = [list(map(int, label.split(' '))) for label in self.table['LABEL'].tolist()]
        if text_only:
            Y.sort(key=len, reverse=True)

        # Bucketing, X & X_len is dummy when text_only==True
        self.X = []
        self.Y = []
        tmp_x, tmp_len, tmp_y = [], [], []
        for x, x_len, y in zip(X, X_lens, Y):
            tmp_x.append(x)
            tmp_len.append(x_len)
            tmp_y.append(y)
            # Half  the batch size if seq too long
            if len(tmp_x) == bucket_size:
                if (bucket_size >= 2) and ((max(tmp_len) > HALF_BATCHSIZE_TIME) or (max([len(y) for y in tmp_y])>HALF_BATCHSIZE_LABEL)):
                    self.X.append(tmp_x[:bucket_size//2])
                    self.X.append(tmp_x[bucket_size//2:])
                    self.Y.append(tmp_y[:bucket_size//2])
                    self.Y.append(tmp_y[bucket_size//2:])
                else:
                    self.X.append(tmp_x)
                    self.Y.append(tmp_y)
                tmp_x, tmp_len, tmp_y = [], [], []
        if len(tmp_x) > 0:
            self.X.append(tmp_x)
            self.Y.append(tmp_y)

    def __getitem__(self, index):
        # Load label
        y = [y for y in self.Y[index]]
        y = target_padding(y, max([len(v) for v in y]))
        if self.text_only:
            return y
        
        # Load acoustic feature and pad
        x = [torch.FloatTensor(np.load(f)) for f in self.X[index]]
        x = pad_sequence(x, batch_first=True)
        return x, y

    def __len__(self):
        return len(self.Y)


def LoadDataset(split, text_only, data_path, batch_size, min_timestep, max_timestep, max_label_len, use_gpu, n_jobs,
                train_set, dev_set, test_set, dev_batch_size, decode_beam_size,**kwargs):
    if split == 'train':
        bs = batch_size
        shuffle = True
        sets = train_set
        drop_too_long = True
    elif split == 'dev':
        bs = dev_batch_size
        shuffle = False
        sets = dev_set
        drop_too_long = True
    elif split == 'test':
        bs = 1 if decode_beam_size>1 else dev_batch_size
        n_jobs = 1
        shuffle = False
        sets = test_set
        drop_too_long = False
    elif split == 'text':
        bs = batch_size
        shuffle = True
        sets = train_set
        drop_too_long = True
    else:
        raise NotImplementedError
        
    ds = LibriDataset(root_path=data_path, sets=sets, min_timestep=min_timestep, max_timestep=max_timestep, text_only=text_only,
                      max_label_len=max_label_len, bucket_size=bs, drop=drop_too_long)
    print(F"{data_path} has {ds.__len__()} batches.")

    return DataLoader(ds, batch_size=1, shuffle=shuffle, drop_last=False, num_workers=n_jobs, pin_memory=use_gpu)


if __name__ == "__main__":
    # total_path = "/home3/zhangzhan/datasets/LibriSpeech/"
    # set_name = "train-clean-100"
    # test = LibriDataset(total_path, [set_name], bucket_size=16)
    # x, y = test.__getitem__(0)
    # print(x.shape, y.shape)
    config = yaml.load(open("dataset/train-clean.yaml", 'r'))
    train_loader = LoadDataset(split="train", **config["trainer"])
    for x, y in train_loader:
        print(x.shape, y.shape)
