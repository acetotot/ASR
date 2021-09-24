from file_wav import get_wav_symbol
import numpy as np
import pandas as pd
from tqdm import tqdm
def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True
print(is_all_chinese("我又g"))

