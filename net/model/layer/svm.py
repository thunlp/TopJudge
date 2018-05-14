from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import json
import os

from net.data_formatter import generate_vector

class svm():
    def __init__(self, config):
        f = open(os.path.join(config.get("data", "svm"), "xf_cut.json"), 'r')
        self.law_content = json.loads(f.readline())
        self.tfidf = joblib.load(os.path.join(config.get("data", "svm"), "cail.tfidf"))
        self.svm = joblib.load(os.path.join(config.get("data", "svm"), "cail_law.model"))
        f = open(os.path.join(config.get("data", "svm"), "law_dict.json"), 'r')
        tmp = json.loads(f.readline())
        self.law_dict = {}
        for key in tmp.keys():
                self.law_dict[tmp[key]] = key
        print("svm model load success")
    # law_content, tfidf, svm, law_dict = init()

    def top2law(self,fact):
        tmp = ''
        for s in fact:
                tmp += ' '.join(s)

        vec = self.tfidf.transform([tmp])
        scores = self.svm.decision_function(vec)
        m = [i for i in range(len(scores[0]))]
        m.sort(reverse = True, key = lambda i : scores[0][i])
        vecs = []
        for i in range(config.getint("data", "top_k")):
            vec, __ = generate_vector(self.law_content[self.law_dict[m[i]]])
            vecs.append(vec)

        return vecs
        # print(scores)
        # print(law_content[law_dict[m[0]]], law_content[law_dict[m[1]]])
        # return law_content[m[0]], law_content[m[1]]
