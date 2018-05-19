from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import json
import os
from torch.autograd import Variable

from net.data_formatter import generate_vector

class svm():
    def __init__(self, config, usegpu):
        print("begin loading svm model")
        f = open(os.path.join(config.get("data", "svm"), "xf_cut.json"), 'r')
        self.law_content = json.loads(f.readline())
        from net.file_reader import transformer
        for i in self.law_content.keys():
            tmp, __ = generate_vector(self.law_content[i], config, transformer)
            if usegpu:
                self.law_content[i] = Variable(tmp.cuda())
            else:
                self.law_content[i] = Variable(tmp)
        self.tfidf = joblib.load(os.path.join(config.get("data", "svm"), "{0}.tfidf".format(config.get("data", "dataset"))))
        self.svm = joblib.load(os.path.join(config.get("data", "svm"), "{0}_law.model".format(config.get("data", "dataset"))))
        # f = open(os.path.join(config.get("data", "svm"), "law_dict.json"), 'r')
        from net.loader import law_dict
        tmp = law_dict.copy()
        print(tmp)
        # f.close()
        self.law_dict = {}
        for key in tmp.keys():
                self.law_dict[tmp[key]] = str([key[0], key[1]])
        print("svm model load success")
    # law_content, tfidf, svm, law_dict = init()

    def top2law(self, config, fact):
        tmp = ''
        for s in fact:
                tmp += ' '.join(s)

        vec = self.tfidf.transform([tmp])
        scores = self.svm.decision_function(vec)
        m = [i for i in range(len(scores[0]))]
        m.sort(reverse = True, key = lambda i : scores[0][i])
        vec = []
        for i in range(config.getint("data", "top_k")):
            vec.append(self.law_content[self.law_dict[m[i]]])
        return vec
        # print(scores)
        # print(law_content[law_dict[m[0]]], law_content[law_dict[m[1]]])
        # return law_content[m[0]], law_content[m[1]]
