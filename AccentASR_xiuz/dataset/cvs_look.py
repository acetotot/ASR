import numpy as np
import pandas as pd
import torchaudio
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from torchaudio.compliance.kaldi import fbank as compute_fbank
import torchaudio
import shutil
from tqdm import tqdm
# f="/data/aidatatang-1505/npy/train-clean/G0105S0120.npy" #2508589
# filepath = "/data/train-aishell-aitang.cvs"
# df=pd.read_csv(filepath,header=0,encoding="gbk")
# path_list = list(df["feature_path"])
# wu = []
# id = []  /data/aishell-2/npy/train-clean/IC0217W0023.npy
# for i,f in tqdm(enumerate(path_list), total=len(path_list)):
#     feature = np.load(f)
#     if feature.shape[-1] != 40:
#         wu.append(f)
#         id.append(i)
#         print(df["features_len"][i], f)


filepath = "/data/test-aishell-aitang.cvs"
df=pd.read_csv(filepath,header=0,encoding="gbk")
path_list = list(df["feature_path"])
for i,f in tqdm(enumerate(path_list), total=len(path_list)):
        try:
            feature = np.load(f)
        except Exception as e:
            print(e, f)
# #


# 单个音频生成npy文件 /data/aishell-2/npy/train-clean/IC0217W0095.npy  /data/aishell-2/npy/train-clean/IC0217W0127.npy /data/aishell-2/npy/train-clean/IC0216W0363.npy
# raw, sr = torchaudio.load("/data-raw/aishell-2/wav/train/C0216/IC0216W0381.wav")
# fbank = compute_fbank(raw, num_mel_bins=40)
# cache_file_path = F"/data/aishell-2/npy/train-clean/IC0216W0381.npy"
# np.save(cache_file_path, fbank)
#
# ########检验是否可load############################
# f = "/data/aishell-2/npy/train-clean/IC0216W0381.npy"
# try:
#     feature = np.load(f)
# except Exception as e:
#     print(e, f)
# #