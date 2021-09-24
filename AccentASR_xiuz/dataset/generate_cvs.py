import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from torchaudio.compliance.kaldi import fbank as compute_fbank
import torchaudio
import shutil
def deleteDuplicatedElementFromList3(listA):
    return sorted(set(listA), key=listA.index)
###########################生成cvs程序######################################
def GetSymbolList():
    '''
    加载拼音符号列表，用于标记符号返回一个列表list类型变量
    '''
    txt_obj = open("/data2/lzl/project/speechz/dict-hanzi.txt", 'r', encoding='gbk')  # 打开文件并读入 拼音是utf-8 汉字表改gbk
    txt_text = txt_obj.read()  #汉字的话改dict 变成汉语表
    txt_lines = txt_text.split('\n')  # 文本分割
    list_symbol = []  # 初始化符号列表
    for i in txt_lines:
        if (i != ''):
            txt_l = i.split('\t')
            list_symbol.append(txt_l[1])##汉字是1，拼音是0
    txt_obj.close()
    list_symbol.append('_')
    return list_symbol



list_symbol = GetSymbolList()#6728个
filename = "/data/aidatatang-1505/train_hanzi.syllable.txt"
txt_obj = open(filename, 'r',encoding='utf-8')  # 打开文件并读入
txt_text_syllable = txt_obj.read()
txt_lines_syllable = txt_text_syllable.split('\n')  # 文本分割
# # # id pin yin
filename = "/data/aidatatang-1505/train_hanzi.wav.txt"
txt_obj = open(filename, 'r',encoding='gbk')  # 打开文件并读入
txt_text_wav = txt_obj.read()
txt_lines_wav = txt_text_wav.split('\n')  # 文本分割
# # # id path/id.wav
#####################求出第一列id列表##################################
id_list = []
for i in txt_lines_syllable:
    if (i != ''):
        txt_l = i.split('	')
        id_list.append(txt_l[0])
print(len(id_list))
# ####################求出第二列拼音文本列表##################################
txt_list = []
label_list = []
wu = []
x=0
for i in txt_lines_syllable:
    if (i != ''):
        txt_l = i.split('	')
        txt_list.append(txt_l[1]) #汉字标签没有空格 所以是txt——l【1】
        feat_out = []
        for s in txt_l[1]:  #汉字标签没有空格 所以是txt——l【1】
            if ('' != s):
                    if s =="薙":
                        s="剃"
                    try:
                        n = list_symbol.index(s)
                    except ValueError:
                        print(s)
                        x+=1
                        n = 8
                        wu.append(s)
                    feat_out.append(n)
        label_list.append(feat_out)
print(x)
print(len(txt_list))
print(len(label_list))
print(deleteDuplicatedElementFromList3(wu))
#####################求出第三/四列音频特征路径、特征长度##################################
fearture_path_list = []
feature_lens_list = []
for i in txt_lines_wav:
    if (i != ''):
        txt_l = i.split('	')
        raw, sr = torchaudio.load(txt_l[1])
        fbank = compute_fbank(raw, num_mel_bins=40)
        cache_file_path = F"/data/aidatatang-1505/npy/train-clean/{txt_l[0]}.npy"
        np.save(cache_file_path, fbank)
        feature_lens = len(fbank)
        fearture_path_list.append(cache_file_path)
        feature_lens_list.append(feature_lens)
print('load feature')
# # #####################求出第五列标签数字列表##################################
df_dict={"utt_id": id_list, "text": txt_list,"feature_path":fearture_path_list,"feature_lens":feature_lens_list,"lable":label_list}
df = pd.DataFrame(df_dict)
# df = df.sort_values(["feature_lens"])
df.to_csv(F"/data/aidatatang-1505/npy/train_hanzi.cvs", index=False)



#tr 正在生成aidatang的文字 train 的cvs





# ##############################################################################
# #####上面是cvs文件得到，下面是分数据集和验证集
# ##############################################################################
#
# def GetMiddleStr(content,startStr,endStr):
#   startIndex = content.index(startStr)
#   if startIndex>=0:
#     startIndex += len(startStr)
#   endIndex = content.index(endStr)
#   return content[startIndex:endIndex]

###############分割数据集##################################
# f1 = open(F"/data/aishell-2/npy/data-clean.cvs",'r') #需要处理的文本
# csvfile = f1.readlines()
# f2 = open(F"/data/aishell-2/npy/train.cvs",'w+') #训练集
# f3 = open(F"/data/aishell-2/npy/dev.cvs",'w+') #验证集如果读取的不是字节那就把b去掉 a是追加 要重新写用w
# for i,line in enumerate(csvfile):
#     if line != '':
#         if i<960000:
#             f2.write(line)
#         if i==960000:
#             f3.write(csvfile[0])
#         if i>=960000:
#             npy_path = GetMiddleStr(line,"/data",".npy")
#             train_path = '/data' + npy_path + '.npy'
#             new_path = train_path.replace('train-clean', 'dev-clean')
#             shutil.move(train_path,new_path[0:29])
#             line = line.replace('train-clean','dev-clean')
#             f3.write(line)
#         if i>962815:
#             break



