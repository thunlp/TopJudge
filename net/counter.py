# coding: UTF-8

import os
import json
import re
import configparser
import pdb

from net.data_formatter import check, generate_vector, format_sentence
from net.file_reader import transformer

configFilePath = "../config/multi_lstm/crit/small.config"
config = configparser.RawConfigParser()
config.read(configFilePath)

# in_path = r"D:\work\law_pre\test\in"
# out_path = r"D:\work\law_pre\test\out"
in_path = r"/data/disk1/private/zhonghaoxi/law/final_data2"
out_path = r"/data/disk1/private/zhonghaoxi/law/data2"
res_path = r"result/count_data"

num_file = 20
num_process = 1

total_cnt = 0

global_crit = {}
global_time = {}
global_law = {}
crit = {}
time = {}
law = {}


def print_res(cnt, law, crit, time, ouf):
    print("total %d" % cnt, file=ouf)
    print("law", file=ouf)
    for x in law.keys():
        print(x, law[x], file=ouf)

    print("\ncrit", file=ouf)
    for x in crit.keys():
        print(x, crit[x], file=ouf)

    print("\ntime", file=ouf)
    for x in time.keys():
        print(x, time[x], file=ouf)


def add(dic, key):
    if not (key in dic.keys()):
        dic[key] = 0
    dic[key] += 1


def count(data, config):
    global total_cnt
    total_cnt += 1

    for x in data["law"]:
        if x[0] < 102 or x[0] > 452:
            continue
        add(law, x)
        add(global_law, x)

    for x in data["time"]["youqi"]:
        add(time, x)
        add(global_time, x)
    if data["time"]["wuqi"]:
        add(time, "wuqi")
        add(global_time, "wuqi")
    if data["time"]["sixing"]:
        add(time, "sixing")
        add(global_time, "sixing")

    for x in data["crit"]:
        add(crit, x)
        add(global_crit, x)


def parse(data):
    # pdb.set_trace()
    res = {}
    res["content"] = format_senetence(data["content"], config)

    res["meta"] = {}
    res["meta"]["law"] = []
    for x in data["meta"]["name_of_law"]:
        res["meta"]["law"].append((x["tiao_num"], x["zhiyi"], x["kuan_num"]))

    res["meta"]["crit"] = []
    for x in data["meta"]["name_of_accusation"]:
        res["meta"]["crit"].append(x)

    res["meta"]["time"] = data["meta"]["term_of_imprisonment"]
    res["meta"]["money"] = data["meta"]["punish_of_money"]
    res["meta"]["criminals"] = data["meta"]["criminals"]

    return res


def draw_out(in_path, out_path, res_path):
    global crit, law, time
    crit = {}
    law = {}
    time = {}

    print(in_path)
    inf = open(in_path, "r")
    ouf = open(out_path, "w")

    cnt = 0
    for line in inf:
        data = json.loads(line)
        if not (check(data, config)):
            continue
        data = parse(data)
        count(data["meta"], config)
        cnt += 1
        data["content"] = generate_vector(data["content"], config, transformer)
        print(json.dumps(data, ensure_ascii=False), file=ouf)
        if cnt % 5000 == 0:
            print(cnt)

    resf = open(res_path, "w")
    print_res(cnt, law, crit, time, resf)


def work(from_id, to_id):
    for a in range(int(from_id), int(to_id)):
        print(str(a) + " begin to work")
        draw_out(os.path.join(in_path, str(a)), os.path.join(out_path, str(a)), os.path.join(res_path, str(a)))
        print(str(a) + " work done")


if __name__ == "__main__":
    work(0, num_file)

    ouf = open("result/count_data/total.txt", "w")

    print_res(total_cnt, global_law, global_crit, global_time, ouf)
