from file_wav import get_wav_symbol

txt_obj = open("/data2/lzl/project/speech/dict.txt", 'r', encoding='UTF-8')  # 打开文件并读入
txt_text = txt_obj.read()
txt_lines = txt_text.split('\n')  # 文本分割
list1_symbol = []  # 初始化符号列表
for i in txt_lines:
    if (i != ''):
        txt_l = i.split('\t')
        list1_symbol.append(txt_l[0])
txt_obj.close()
list1_symbol.append('_')
# if 'shang2' in list1_symbol:
#     print('1')

filename = "/data/aidatatang-1505/train.syllable.txt"
txt_obj = open(filename, 'r', encoding='utf-8')  # 打开文件并读入
txt_text = txt_obj.read()
txt_lines = txt_text.split('\n')  # 文本分割
dic_symbol_list = {}  # 初始化字典
list_symbolmark = []  # 初始化symbol列表
for i in txt_lines:
    if (i != ''):
        txt_l = i.split("\t")
        dic_symbol_list[txt_l[0]] = txt_l[1:]  # 字典{‘对应序号’:['er4','jin1', 'tian1', 'mei2', 'ke4', 'ma5', 'hai2', 'he2', 'li3', 'xia2', 'liao2', 'tian1']}
        list_symbolmark.append(txt_l[0])

not_in = []
for i in range(len(dic_symbol_list)-1):
    list_symbol = dic_symbol_list[list_symbolmark[i]]
    for i in list_symbol:
        if ('' != i):
            if i not in list1_symbol:
                print(i)
                not_in.append(i)


filename = "/data/aidatatang-1505/dev.syllable.txt"
txt_obj = open(filename, 'r', encoding='utf-8')  # 打开文件并读入
txt_text = txt_obj.read()
txt_lines = txt_text.split('\n')  # 文本分割
dic_symbol_list = {}  # 初始化字典
list_symbolmark = []  # 初始化symbol列表
for i in txt_lines:
    if (i != ''):
        txt_l = i.split("\t")
        dic_symbol_list[txt_l[0]] = txt_l[1:]  # 字典{‘对应序号’:['er4','jin1', 'tian1', 'mei2', 'ke4', 'ma5', 'hai2', 'he2', 'li3', 'xia2', 'liao2', 'tian1']}
        list_symbolmark.append(txt_l[0])

for i in range(len(dic_symbol_list)-1):
    list_symbol = dic_symbol_list[list_symbolmark[i]]
    for i in list_symbol:
        if ('' != i):
            if i not in list1_symbol:
                print(i)

