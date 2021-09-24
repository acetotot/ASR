import torchaudio
from torchaudio.compliance.kaldi import fbank as compute_fbank

import editdistance as ed
def cal_cer_batch_hanzi(pred, label): #编辑次数除以总单词数
    eds = [min(1.0, float(ed.eval([char for char in p], [char for char in l])) / len(l)) for p, l in zip(pred, label)]
    return sum(eds) / len(eds)
def cal_cer_batch(pred, label): #编辑次数除以总单词数
    eds = [min(1.0, float(ed.eval(p, l)) / len(l)) for p, l in zip(pred, label)]
    return sum(eds) / len(eds)
a = "so re ahh"
b = "si ee hh"

cer = cal_cer_batch(a,b)
print(cer)