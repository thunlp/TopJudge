import json
import thulac
import os

from net.parser import ConfigParser
from net.data_formatter import check_sentence

out_path = "/data/zhx/law/data/cail"

ouf = []
for a in range(0, 20):
    ouf.append(open(os.path.join(out_path, str(a)), "w"))

cutter = thulac.thulac(model_path=r"/data/zhx/thulac/models", seg_only=True, filt=False)

config = ConfigParser("/home/zhx/law_pre/config/default_config.config")

file_list = ["/data/zhx/contest/small/data_test.json", "/data/zhx/contest/small/data_train.json",
             "/data/zhx/contest/small/data_valid.json"]

cnt = 0


def cut(s):
    data = cutter.cut(s)
    result = []
    first = True
    for x, y in data:
        if x == " ":
            continue
        result.append(x)
    return result


for file_name in file_list:
    f = open(file_name, "r", encoding="utf8")
    for line in f:
        try:
            data = json.loads(line)
            result = {}
            result["meta"] = {}
            result["meta"]["crit"] = data["meta"]["accusation"]
            result["meta"]["time"] = {}
            result["meta"]["time"]["youqi"] = [data["meta"]["term_of_imprisonment"]["imprisonment"]]
            result["meta"]["time"]["guanzhi"] = []
            result["meta"]["time"]["huanxing"] = []
            result["meta"]["time"]["juyi"] = []
            result["meta"]["time"]["sixing"] = data["meta"]["term_of_imprisonment"]["death_penalty"]
            result["meta"]["time"]["wuqi"] = data["meta"]["term_of_imprisonment"]["life_imprisonment"]
            result["meta"]["criminals"] = data["meta"]["criminals"]
            result["meta"]["law"] = []
            for x in data["meta"]["relevant_articles"]:
                result["meta"]["law"].append((x, 0, 0))

            fact = cut(data["fact"])
            res = [[]]
            for x in fact:
                if x == "。":
                    res.append([])
                else:
                    res[-1].append(x)
            if not (check_sentence(res, config)):
                continue
            result["fact"] = res

            cnt += 1
            op = cnt % 20
            print(json.dumps(result, ensure_ascii=False), file=ouf[op])
            if cnt % 5000 == 0:
                print(file_name, cnt)
        except Exception as e:
            gg

print(cnt)
