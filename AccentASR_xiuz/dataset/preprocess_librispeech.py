import glob
import os
from tqdm import tqdm
from g2p_en import G2p
from utils.text_utils import *
from utils.feature_utils import *
import pandas as pd
from joblib import Parallel, delayed
import pickle


def get_saved_name(filename):
    return filename.replace(".flac", "")


def cache_features_and_get_len_df(dataset_path):
    wav_list = glob.glob(F"{dataset_path}/*/*/*.flac")
    tr_x = Parallel(n_jobs=4)(delayed(extract_feature)(str(file), feature="mfcc", dim=13,
                                                       cmvn=True, delta=True, delta_delta=True,
                                                       save_feature=get_saved_name(file))
                              for file in tqdm(wav_list))
    tr_x_dict = pd.DataFrame(tr_x)
    return tr_x_dict


def generate_trans(dataset_path):
    text_list = glob.glob(F"{dataset_path}/*/*/*.txt")
    wav_id_list = []
    wav_path_list = []
    word_trans_list = []
    phone_trans_list = []
    g2p = G2p()
    for text_path in tqdm(text_list):
        with open(text_path, "r") as f:
            text = f.readlines()
            for l in text:
                l = l.replace("\n", "").split(" ")
                wav_id = l[0]
                word_trans = ""
                phone_trans = ""
                for w in l[1:]:
                    if w != " ":
                        word_trans += F"{w} "
                        phone_trans += F"{g2p_list_to_string(g2p(w))} "
                wav_id_list.append(wav_id)
                wav_id_s = wav_id.split("-")
                wav_path_list.append(F"{dataset_path}/{wav_id_s[0]}/{wav_id_s[1]}/{wav_id}.npy")
                word_trans_list.append(word_trans[:-1])
                phone_trans_list.append(phone_trans[:-1])
    return {"ID":wav_id_list, "TRANS":word_trans_list, "PHONE_TRANS":phone_trans_list, "PATH":wav_path_list}


if __name__ == "__main__":
    if os.path.exists("mapping.pkl"):
        dump = False
        with open('mapping.pkl', 'rb') as fp:
            encode_table = pickle.load(fp)
    else:
        dump = True
        encode_table = None
    total_path = "/home5/zhangzhan/datasets/LibriSpeech/"
    # sets = ["train-clean-100", "train-clean-360", "train-other-500", "dev-clean", "dev-other"]
    # sets = ["dev-clean", "dev-other", "train-clean-100", "train-clean-360", "train-other-500"]
    sets = ["test-clean"]
    # sets = ["train-clean-100", "train-clean-360", "dev-clean"]
    # sets = ["dev-clean"]
    for set_name in sets:
        print("*"*50)
        print("Currently processing ", set_name)
        dataset_path = os.path.join(total_path, set_name)
        len_df = cache_features_and_get_len_df(dataset_path)
        trans_df = pd.DataFrame(generate_trans(dataset_path))
        final_df = len_df.set_index("ID").join(trans_df.set_index("ID"))
        final_df = final_df.sort_values(["LEN"])
        text_list = list(final_df["PHONE_TRANS"])
        print(encode_table)
        tr_y, encode_table = encode_text_list(text_list, encode_table=encode_table)

        if dump:
            # 39 Phonemes + Blank
            assert len(encode_table) == 40
            with open("mapping.pkl", "wb") as fp:
                pickle.dump(encode_table, fp)
            dump = False
        final_df["LABEL"] = tr_y
        final_df.to_csv(F"{set_name}.csv")


