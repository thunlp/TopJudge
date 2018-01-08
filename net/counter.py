# coding: UTF-8

import os
import json
import re
from data_formatter import check, analyze_time

# in_path = r"D:\work\law_pre\test\in"
# out_path = r"D:\work\law_pre\test\out"
in_path = r"/data/disk1/private/zhonghaoxi/law/data"
out_path = r"/disk/mysql/law_data/count_data"
mid_text = u"  _(:з」∠)_  "
title_list = ["docId", "caseNumber", "caseName", "spcx", "court", "time", "caseType", "bgkly", "yuanwen", "document",
              "cause", "docType", "keyword", "lawyer", "punishment", "result", "judge"]

accusation_file = r"../data_processor/accusation_list2.txt"
accusation_f = open(accusation_file, "r", encoding='utf8')
accusation_list = json.loads(accusation_f.readline())
# for a in range(0, len(accusation_list)):
#    accusation_list[a] = accusation_list[a].replace("[", "").replace("]", "")
# accusation_list = []
# for line in accusation_f:
#    accusation_list.append(line[:-1])

num_file = 20
num_process = 1

total_cnt = 0

crit_list = []
for a in range(0, len(accusation_list)):
    crit_list.append(0)
time_dict = {}


def analyze_times(data):
    x = analyze_time(data, None)
    if not (x in time_dict):
        time_dict[x] = 0
    time_dict[x] += 1


def analyze_crit(data):
    if len(data) == 0:
        return
    for x in data:
        crit_list[x] += 1


law_list = {}


def analyze_law(data):
    for x in data:
        print(x)
        gg


def count(data):
    global total_cnt
    total_cnt += 1

    analyze_crit(data["crit"])
    analyze_times(data["time"])
    analyze_law(data["law"])


def draw_out(in_path, out_path):
    print(in_path)
    inf = open(in_path, "r")

    cnt = 0
    for line in inf:
        data = json.loads(line)
        if not (check(data, None)):
            continue
        count(data["meta"])
        cnt += 1
        if cnt % 500000 == 0:
            print(cnt)


def work(from_id, to_id):
    for a in range(int(from_id), int(to_id)):
        print(str(a) + " begin to work")
        draw_out(os.path.join(in_path, str(a)), os.path.join(out_path, str(a)))
        print(str(a) + " work done")


if __name__ == "__main__":
    work(0, 20)

    ouf = open("result/law_result.txt", "w")
    data = {}
    print(total_cnt)
    gg = 0
    for a in range(0, len(crit_list)):
        # if crit_list[a] > 1000:
        # print(accusation_list[a], a, crit_list[a],file=ouf)
        gg += crit_list[a]
    print(gg)
    data["total"] = total_cnt

    data["crit"] = crit_list
    data["law"] = law_list
    data["time"] = time_dict

    # for x in time_dict.keys():
    #    print(x, time_dict[x])
    # print(json.dumps(data), file=ouf)

    for x in law_list.keys():
        print(x, file=ouf)
