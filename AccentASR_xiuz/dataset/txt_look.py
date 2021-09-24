def deleteDuplicatedElementFromList3(listA):
    return sorted(set(listA), key=listA.index)

###扩充词表汉字程序
# txt_obj = open("/data2/lzl/project/speechz/hanzi.txt", 'r', encoding='gbk')  # 打开文件并读入
# txt_text = txt_obj.read()
# txt_lines = txt_text.split('\n')  # 文本分割
# f = open("/data2/lzl/project/speechz/dict.txt",'w+',encoding='gbk') #训练集
# hanzi_list = []
# for i in txt_lines:
#     if (i != ''):
#         txt_line = i.split(",")
#         hanzi = list(txt_line[1])
#         hanzi_list.extend(hanzi)#extend 直接写 不用返回给a=a.extend
# hanzi_list = deleteDuplicatedElementFromList3(hanzi_list)
# for i,hanzi in enumerate(hanzi_list):
#    f.write(str(i+1)+'\t'+hanzi+"\n")
#
txt_obj = open("/data/train-aishell-aitang.cvs", 'r', encoding='gbk')  # 打开文件并读入
txt_text = txt_obj.read()
txt_lines = txt_text.split('\n')  # 文本分割
l = len(txt_lines)
a = txt_lines[2]
b = txt_lines[-2]
c = txt_lines[3]