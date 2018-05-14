from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import json
import os

from net.data_formatter import generate_vector

class svm():
    def __init__(self, config):
        print("begin loading svm model")
        f = open(os.path.join(config.get("data", "svm"), "xf_cut.json"), 'r')
        self.law_content = json.loads(f.readline())
        from net.file_reader import transformer
        for i in self.law_content.keys():
            self.law_content[i], __ = generate_vector(self.law_content[i], config, transformer)
        self.tfidf = joblib.load(os.path.join(config.get("data", "svm"), "cail.tfidf"))
        self.svm = joblib.load(os.path.join(config.get("data", "svm"), "cail_law.model"))
        f = open(os.path.join(config.get("data", "svm"), "law_dict.json"), 'r')
        tmp = json.loads(f.readline())
        f.close()
        self.law_dict = {}
        for key in tmp.keys():
                self.law_dict[tmp[key]] = key
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
