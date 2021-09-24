# -*- coding: utf-8 -*-
import os
# coding: utf-8
import codecs
import glob
from xpinyin import Pinyin
p = Pinyin()
def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True
print("start globing")
wav_name = glob.glob("/data-raw/aidatatang-1505/1505/dst/data/*/*/*/*.txt")
# wav_name = glob.glob("/data-raw/aidatatang-1505/1505/dst/data/category1/G0002/*/*.txt")

print(len(wav_name)) #一共1623123个wav

f1 = open(r"/data/aidatatang-1505/data_hanzi.syllable.txt", 'w',encoding='utf-8')  # yinbiao
f3 = open(r"/data/aidatatang-1505/wuzyh_hanzi.wav.txt", 'w',encoding='utf-8')  # yinpin
f2 = open(r"/data/aidatatang-1505/data_hanzi.wav.txt", 'w',encoding='utf-8')  # yinpin
chinese_toknes = ["~","“","”",".","。",".", "，",",","!", "！", ":","：","？","?","…","、","；",";", " "]
for a in wav_name:
    with codecs.open(a, encoding='utf-8') as f:
        line = f.readline().strip()
        line = line.replace("嚒", "么")
        line = line.replace("塆", "湾")
        for t in chinese_toknes:
            line = line.replace(t, "")
        assert len(line) > 0
        if is_all_chinese(line)==True:
            id = a[-14:-4]
            # line = p.get_pinyin(line, '	', tone_marks='numbers')
            f1.write(id+"\t")
            f1.write(line + '\n')
            f2.write(id+"\t")
            f2.write(a[:-3]+'wav'+ '\n')
        else:
            f3.write(line+"\n")























#['/data-raw/aidatatang-1505/1505/dst/data/category3/G2020', '/data-raw/aidatatang-1505/1505/dst/data/categ  ]                     ory3/G0788']
#1.glob("/data-raw/aidatatang-1505/1505/dst/data/*/*/*/*.txt")得到一个txt文本路径列表
#2.for 整个文本路径list列表，for i in list 打开每个i路径下的对应文本，并读取第一行内容
#3.判断内容是否为全中文，是的话就继续4以及5

#4.是中文就把i的对应序号部分写入新的syllble。txt 再写tab  再把这个中文内容翻译成拼音写入 再写换行
#5.是中文就把i的对应序号部分写入新的wave。txt 再写tab 再把i最后的。txt改成wav写入 再写换行
#6.遍历一遍后 结束
#7.通过fileprocess进行分文本处理
# [/data2/lzl/1505/T0G01.txt,/data2/lzl/1505/T0G02.txt,/data2/lzl/1505/T0G03.txt]
# open /data2/lzl/1505/T0G01.txt
# read a=/data2/lzl/1505/T0G01.txt = 哈哈哈哈你
# 新建syllble.txt文本
#     写G01 xpinyin（a）
# 新建wave。txt文本
#     写G01 /data2/lzl/1505/T0G01.wav
