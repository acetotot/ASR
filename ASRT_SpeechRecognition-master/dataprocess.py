import re
from xpinyin import Pinyin
p = Pinyin()
filename = "/data-raw/aishell-2/transcript/transcript_aishell2.txt"
txt_obj = open(filename, 'r',encoding='utf-8')  # 打开文件并读入
txt_text = txt_obj.read()
txt_lines = txt_text.split('\n')  # 文本分割
list_wavmark = []
list_pinyin = []
def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True
with open("/data/aishell-2/train.syllable.txt", "w", encoding='utf-8') as f:
    for i in txt_lines:
        if (i != ''):
            txt_l=i.split('	')
            if is_all_chinese(txt_l[1])==True:
                f.write(txt_l[0]+'	')
                list_wavmark.append(txt_l[0])
                txt_l[1] = p.get_pinyin(txt_l[1],'	',tone_marks='numbers')
                f.write(txt_l[1] + '\n')
            # list_pinyin.append(txt_l[1])
txt_obj.close()
f.close()
print(len(list_wavmark))
with open("/data/aishell-2/train.wav.txt", "w", encoding='utf-8') as f:
    for wav_id in list_wavmark:
        f.write(wav_id +"\t")
        s = wav_id[1:6]
        f.write('/data-raw/aishell-2/wav/train/'+s+'/'+ wav_id + '.wav'+'\n')#path要换成音频路径
f.close()


