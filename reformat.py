import os
import json
import thulac

from net.parser import ConfigParser
from net.data_formatter import check_sentence

in_path = "/data/zhx/law/siftData"
out_path = "/data/zhx/law/data/cail"

ouf = []
for a in range(0, 20):
    ouf.append(open(os.path.join(out_path, str(a)), "w"))

cutter = thulac.thulac(model_path=r"/data/zhx/thulac/models", seg_only=True, filt=False)

config = ConfigParser("/home/zhx/law_pre/config/default_config.config")


def cut(s):
    data = cutter.cut(s)
    result = []
    first = True
    for x, y in data:
        if x == " ":
            continue
        result.append(x)
    return result


cnt = 0
for a in range(0, 58):
    inf = open(os.path.join(in_path, "clean_result_%d.json" % a), "r")
    for line in inf:
        try:
            data = json.loads(line)
            result = {}
            result["meta"] = {}
            result["meta"]["crit"] = data["meta"]["accusation"]
            result["meta"]["time"] = {}
            result["meta"]["time"]["youqi"] = [data["meta"]["term_of_imprisonment"]["imprisonment"]]
            result["meta"]["time"]["guanzhi"] = [data["meta"]["term_of_imprisonment"]["control"]]
            result["meta"]["time"]["huanxing"] = [data["meta"]["term_of_imprisonment"]["probation"]]
            result["meta"]["time"]["juyi"] = [data["meta"]["term_of_imprisonment"]["detention"]]
            result["meta"]["time"]["sixing"] = data["meta"]["term_of_imprisonment"]["death_penalty"]
            result["meta"]["time"]["wuqi"] = data["meta"]["term_of_imprisonment"]["life_imprisonment"]
            result["meta"]["criminals"] = data["meta"]["criminals"]
            result["meta"]["law"] = []
            for x in data["meta"]["relevant_articles"]:
                result["meta"]["law"].append((x["article"], x["option"], x["section"]))

            fact = cut(data["fact"])
            res = [[]]
            for x in fact:
                if x == "。":
                    res.append([])
                else:
                    res[-1].append(x)
            if not (check_sentence(res, config)):
                continue
            result["fact"]=res

            cnt += 1
            op = cnt % 20
            print(json.dumps(result, ensure_ascii=False), file=ouf[op])
            if cnt % 5000 == 0:
                print(a,cnt)
        except Exception as e:
            gg
