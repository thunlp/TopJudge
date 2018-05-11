# coding:utf-8
import numpy as np
import json
import pickle


class word2vec:
    word_num = 0
    vec_len = 0
    word2id = None
    vec = None

    def __init__(self, word_dic="/data/zhx/law/word2vec/word2id.pkl",
                 vec_path="/data/zhx/law/word2vec/vec_nor.npy"):
        print("begin to load word embedding")
        f = open(word_dic, "rb")
        (self.word_num, self.vec_len) = pickle.load(f)
        self.word2id = pickle.load(f)
        f.close()
        self.vec = np.load(vec_path)
        print("load word embedding succeed")

    def load(self, word):
        try:
            return self.vec[self.word2id[word]].astype(dtype=np.float32)
        except:
            return self.vec[self.word2id['UNK']].astype(dtype=np.float32)


if __name__ == "__main__":
    a = word2vec()
    print(a.vec_len)
    print(a.load('，'))
