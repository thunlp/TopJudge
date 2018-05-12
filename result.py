import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p')
args = parser.parse_args()

path = args.path
name = path.split("/")[-1]

result = {
    "crit": {"acc": 0, "mp": 0, "mr": 0, "f1": 0},
    "law": {"acc": 0, "mp": 0, "mr": 0, "f1": 0},
    "time": {"acc": 0, "mp": 0, "mr": 0, "f1": 0}
}

map_list = {
    0: "acc",
    1: "mp",
    2: "acc",
    3: "mr",
    4: "acc",
    5: "f1"
}

cnt = 0
try:
    while True:
        cnt += 1
        for task in result.keys():
            f = open(os.path.join(path, "%d-%s" % (cnt, task)), "r")
            nowv = {}
            for a in range(0, 6):
                value = float(f.readline()[:-1].split("\t")[-1])
                nowv[map_list[a]] = value
            if nowv["f1"] > result[task]["f1"]:
                for x in nowv.keys():
                    result[task][x] = nowv[x]
except Exception as e:
    print(e)

f = open("result/%s" % name, "w")
for x in ["law","crit","time"]:
    print(x,[result[x]["acc"],result[x]["mp"],result[x]["mr"],result[x]["f1"]], file=f)
    print(x,[result[x]["acc"],result[x]["mp"],result[x]["mr"],result[x]["f1"]])

f.close()
