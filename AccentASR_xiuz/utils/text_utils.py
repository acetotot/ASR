import os
import pickle
import torch
import numpy as np
import editdistance as ed
# from g2p_en import G2p
import sentencepiece as spm
EOS = 1


def GetSymbolList():
    txt_obj = open("/data2/lzl/project/speech-xiuz/dict.txt", 'r', encoding='utf-8')  # 打开文件并读入 hanzi改成gbk
    txt_text = txt_obj.read()
    txt_lines = txt_text.split('\n')  # 文本分割
    list_symbol = []  # 初始化符号列表 <PAD> <SOS> <EOS>
    for i in txt_lines:
        if (i != ''):
            txt_l = i.split('\t')
            list_symbol.append(txt_l[0])
    txt_obj.close()
    list_symbol.append('_')
    return list_symbol

def translate_to_string(list):
    list_symbol = GetSymbolList()
    txt_list = []
    for i in list:
        if int(i) < len(list_symbol):
            txt_list.append(list_symbol[int(i)])
    return txt_list



if __name__ == "__main__":
    a = ["34",'45','98']
    a=translate_to_string(a)
    print(a)










#
class Mapper:
    def __init__(self, spm_model):
        sp = spm.SentencePieceProcessor(model_file=spm_model)
        self.sp = sp

    def encode_sentence(self, s):
        return self.sp.encode(s)

    def translate_to_string(self, s):
        if type(s) == torch.Tensor:
            s = s.cpu().numpy().tolist()
        return self.sp.decode(s)

    def translate_batch_to_strings(self, b):
        if type(b) == torch.Tensor:
            b = b.cpu().numpy()
        strings = []
        for s in b:
            strings.append(self.translate_to_string(s.tolist()))
        return strings


# class PhonemeMapper:
#     def __init__(self, vocab_path=None):
#         # Find mapping
#         if vocab_path is None:
#             vocab_path = 'dataset/mapping.txt'
#         self.mapping = self.load_vocab(vocab_path)
#         self.r_mapping = {v: k for k, v in self.mapping.items()}
#         self.g2p = G2p()
#
#     @staticmethod
#     def load_vocab(vocab_file):
#         unit2idx = {}
#         with open(os.path.join(vocab_file), 'r', encoding='utf-8') as v:
#             for line in v:
#                 unit, idx = line.strip().split()
#                 unit2idx[unit] = int(idx)
#         return unit2idx
#
#     def get_dim(self):
#         return len(self.mapping)
#
#     @staticmethod
#     def g2p_list_to_string(phone_list):
#         s = ""
#         for p in phone_list:
#             p = p.replace("0", "").replace("1", "").replace("2", "")
#             if p != " " and p != "'":
#                 s += F"{p} "
#         return s[:-1]
#
#     def g2p_string(self, word):
#         """
#         get phone string of a certain word or a sentence
#         :param word: english word
#         :return: phone string split by space
#         """
#         text = self.g2p_list_to_string(self.g2p(word))
#         return text
#
#     def encode_sentence(self, s):
#         """
#         encode a sentence to int label
#         :param word: english word
#         :return: int label list
#         """
#         text = self.g2p_string(s)
#         label = self.encode_phonemes(text)
#         return label
#
#     def encode_phonemes(self, text):
#         """
#         encode a space split text into corresponding label
#         :param text:
#         :return:
#         """
#         label = []
#         for t in text.split(" "):
#             if t != "":
#                 label.append(self.mapping[t.upper()])
#         return label
#
#     def translate_batch(self, seq_batch, return_string=False):
#         seq_batch = seq_batch.cpu()
#         new_seq_batch = []
#         for seq in seq_batch:
#             new_seq_batch.append(self.translate(seq, return_string))
#         return new_seq_batch
#
#     def translate(self, seq, return_string=False):
#         new_seq = []
#         last_char = None
#         for c in seq:
#             c = c.cpu().item()
#             if c != 0 and c != last_char:
#                 new_seq.append(self.r_mapping[c])
#             last_char = c
#         if return_string:
#             new_seq = ' '.join(new_seq)
#         return new_seq


def max_decode_last_dim(logits):
    # logits n*c*s
    # prob = torch.softmax(logits, dim=1)
    probs = logits
    seq = probs.argmax(dim=-1)
    return seq


def cal_cer_batch(pred, label): #编辑次数除以总单词数
    eds = [min(1.0, float(ed.eval(p, l)) / len(l)) for p, l in zip(pred, label)]
    return sum(eds) / len(eds)
def cal_cer_batch_hanzi(pred, label): #编辑次数除以总单词数
    eds = [min(1.0, float(ed.eval([char for char in p], [char for char in l])) / len(l)) for p, l in zip(pred, label)]
    return sum(eds) / len(eds)
def cal_wer_batch(pred, label):
    eds = [min(1.0, float(ed.eval(p.split(' '), l.split(' '))) / len(l.split(' '))) for p, l in zip(pred, label)]
    return sum(eds) / len(eds)




