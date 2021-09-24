#############################合并两个txt##########################################
# file1 = "/data/dev.syllable.txt"
# file2 = "/data/aishell-2/test.syllable.txt"
#
# f1 = open(file1, 'a+', encoding='utf-8')
# with open(file2, 'r', encoding='utf-8') as f2:
#     # f1.write('\n')
#     for i in f2:
#         if i!="":
#             f1.write(i)

#############################合并两个数据集的cvs，sort。##############################
import pandas as pd

filepath1="/data/aishell-2/npy/dev_hanzi.cvs"
filepath2="/data/aidatatang-1505/npy/dev_hanzi.cvs"
df1=pd.read_csv(filepath1,header=0,encoding="utf-8") #hanzi改utf-8 pinyin改gbk
df2=pd.read_csv(filepath2,header=0,encoding="utf-8")
df =pd.concat([df1,df2])
df = df.sort_values(["feature_lens"])
# df.to_csv(F"/data/train_hanzi.cvs", index=False)
# print(df)
#########################################删除cvs行数写入新文件#######################################
# f ="/data/train-aishell-aitang.cvs"
# df=pd.read_csv(f,header=0,encoding="gbk")
# cf = df [:-1947]#需要剩下的行数
#在写入cvs
df.to_csv(F"/data/dev_hanzi.cvs", index=False)
print(df)
#########################################删除txt行数写入新文件#######################################
# file1 = "/data/dev.syllable.txt"
# txt_obj = open("/data/train-aishell-aitang.cvs", 'r', encoding='gbk')  # 打开文件并读入
# txt_text = txt_obj.read()
# txt_lines = txt_text.split('\n')  # 文本分割

