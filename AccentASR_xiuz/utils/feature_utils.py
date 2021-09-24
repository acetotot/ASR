import librosa
import numpy as np
import os
from operator import itemgetter
import torch


def extract_feature(input_file, dataset="librispeech", feature='fbank', dim=40,
                    cmvn=True, delta=False, delta_delta=False,
                    window_size=25, stride=10, save_feature=None):
    """
    Acoustic Feature Extraction
    :param input_file: str, audio file path
    :param dataset: this decides how to map input_file to id
    :param feature:  str, fbank or mfcc
    :param dim:  int, dimension of feature
    :param cmvn:  bool, apply CMVN on feature
    :param delta:
    :param delta_delta:
    :param window_size:  int, window size for FFT (ms)
    :param stride:  int, window stride for FFT
    :param save_feature:  str, if given, store feature to the path and return {ID:file_id, LEN:len(feature)}
    :return:  {ID:file_id, LEN:len(feature)} or acoustic features with shape (time step, dim)
    """
    y, sr = librosa.load(input_file, sr=None)
    ws = int(sr*0.001*window_size)
    st = int(sr*0.001*stride)
    if feature == 'fbank':  # log-scaled
        feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=dim,
                                              n_fft=ws, hop_length=st)
        feat = np.log(feat+1e-6)
    elif feature == 'mfcc':
        feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=dim, n_mels=26,
                                    n_fft=ws, hop_length=st)
        # Energy
        feat[0] = librosa.feature.rms(y, hop_length=st, frame_length=ws)

    else:
        raise ValueError('Unsupported Acoustic Feature: '+feature)

    feat = [feat]
    if delta:
        feat.append(librosa.feature.delta(feat[0]))

    if delta_delta:
        feat.append(librosa.feature.delta(feat[0],order=2))
    feat = np.concatenate(feat,axis=0)
    if cmvn:
        feat = (feat - feat.mean(axis=1)[:,np.newaxis]) / (feat.std(axis=1)+1e-16)[:,np.newaxis]
    if save_feature is not None:
        tmp = np.swapaxes(feat,0,1).astype('float32')
        np.save(save_feature, tmp)
        if dataset == "librispeech":
            file_id = os.path.basename(input_file).split(".")[0]
        elif dataset == "timit":
            file_id = "-".join(input_file.strip(".wav").split("/")[-3:])
        else:
            file_id = input_file
        return {"ID":file_id, "LEN":len(tmp)}
    else:
        return np.swapaxes(feat, 0, 1).astype('float32')


def encode_text_list(input_list, encode_table=None, return_detail=False):
    """
    Encode a list of text to a list of int
    :param input_list: List of text
    :param encode_table: If given, use this table to encode text; else, use data to generate table.
    :param return_detail: Whether to return the number of each encoded element
    :return: List of int and encode table
    """
    if encode_table is None:
        table = {}
        for target in input_list:
            for t in target.split(" "):
                if t not in table:
                    table[t] = 1
                else:
                    table[t] += 1
        sorted_table = sorted(table.items(), key=itemgetter(1), reverse=True)
        print("The frequency details of phonemes:")
        print("*"*50)
        for i, (k, v) in enumerate(sorted_table):
            print("%2s" % i, "%2s" % k, "%5d" % v)
        print("*"*50)
        all_tokens = [k for k,v in sorted_table]
        table = {"<blk>":0}
        for tok in all_tokens:
            table[tok] = len(table)
    else:
        table = encode_table
    output_list = []
    for target in input_list:
        tmp = []
        for t in target.split(" "):
            tmp.append(table[t])
        output_list.append(" ".join(map(str, tmp)))
    if return_detail and encode_table is None:
        return output_list, table, sorted_table
    else:
        return output_list, table



def zero_padding(x, pad_len):
    """
    Feature Padding Function
    :param x:  list, list of np.array
    :param pad_len:  int, length to pad (0 for max_len in x)
    :return:  np.array with shape (len(x),pad_len,dim of feature)
    """
    features = x[0].shape[-1]
    if pad_len is 0:
        pad_len = max([len(v) for v in x])
    new_x = np.zeros((len(x), pad_len, features))
    for idx, ins in enumerate(x):
        new_x[idx, :min(len(ins), pad_len),:] = ins[:min(len(ins), pad_len), :]
    return new_x


def target_padding(y, max_len):
    """
    Target Padding Function
    :param y:  list, list of int
    :param max_len:  int, max length of output (0 for max_len in y)
    :return:  np.array with shape (len(y),max_len)
    """
    if max_len is 0:
        max_len = max([len(v) for v in y])
    new_y = np.zeros((len(y), max_len), dtype=int)
    for idx, label_seq in enumerate(y):
        new_y[idx, :len(label_seq)] = np.array(label_seq)
    return new_y


def int_to_tensor(x):
    x = np.asarray(x)
    x = torch.from_numpy(x)
    return x