import codecs
f1 = open(r"/data/aidatatang-1505/data_hanzi.wav.txt",'rb') #需要处理的文本
f2 = open(r"/data/aidatatang-1505/train_hanzi.wav.txt",'ab') #训练集
f3 = open(r"/data/aidatatang-1505/dev_hanzi.wav.txt",'ab') #验证集如果读取的不是字节那就把b去掉 a是追加 要重新写用w
i=-1
while True:
    line = f1.readline()
    i+=1
    if i<1550000:
        f2.write(line)
    if i>=1550000:
        f3.write(line)
    if i>1557170:
        break
f1 = open(r"/data/aidatatang-1505/data_hanzi.syllable.txt",'rb') #需要处理的文本
f2 = open(r"/data/aidatatang-1505/train_hanzi.syllable.txt",'ab') #训练集
f3 = open(r"/data/aidatatang-1505/dev_hanzi.syllable.txt",'ab') #验证集如果读取的不是字节那就把b去掉 a是追加 要重新写用w
i=-1
while True:
    line = f1.readline()
    i+=1
    if i<1550000:
        f2.write(line)
    if i>=1550000:
        f3.write(line)
    if i>1557170:
        break
# with codecs.open("/data/aidatatang-1505/data.wav.txt", encoding='utf-8') as f:
#     txt_text = f.read()
#     txt_lines = txt_text.split('\n')  #
# print(len(txt_lines))
# with codecs.open("/data/aidatatang-1505/data.syllable.txt", encoding='utf-8') as f:
#     txt_text = f.read()
#     txt_lines = txt_text.split('\n')  #
# print(len(txt_lines))
