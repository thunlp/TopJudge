# coding: UTF-8

import os
import json
import re
from data_formatter import check, analyze_time, analyze_law1, analyze_law2, analyze_crit
import configparser
from utils import get_num_classes

configFilePath = "../config/multi_lstm/crit/small.config"
config = configparser.RawConfigParser()
config.read(configFilePath)

# in_path = r"D:\work\law_pre\test\in"
# out_path = r"D:\work\law_pre\test\out"
in_path = r"/data/disk1/private/zhonghaoxi/law/data"
out_path = r"result/count_data"

num_file = 20
num_process = 1

total_cnt = 0

global_crit = []
global_time = []
global_law = []
crit = []
time = []
law = []


def print_res(law, crit, time, ouf):
    print("law", file=ouf)
    for a in range(0, len(law)):
        print(a, law[a], file=ouf)

    print("\ncrit", file=ouf)
    for a in range(0, len(crit)):
        print(a, crit[a], file=ouf)

    print("\ntime", file=ouf)
    for a in range(0, len(time)):
        print(a, time[a], file=ouf)


def count(data, config):
    global total_cnt
    total_cnt += 1

    a = analyze_crit(data["crit"], config)
    b = analyze_time(data["time"], config)
    c = analyze_law1(data["law"], config)
    crit[a] += 1
    global_crit[a] += 1
    time[b] += 1
    global_time[b] += 1
    law[c] += 1
    global_law[c] += 1


def draw_out(in_path, out_path):
    global crit,time,law
    print(in_path)
    inf = open(in_path, "r")
    crit = []
    time = []
    law = []
    for a in range(0, get_num_classes("crit")):
        crit.append(0)
    for a in range(0, get_num_classes("time")):
        time.append(0)
    for a in range(0, get_num_classes("law1")):
        law.append(0)

    cnt = 0
    for line in inf:
        data = json.loads(line)
        if not (check(data, config)):
            continue
        count(data["meta"], config)
        cnt += 1
        if cnt % 500000 == 0:
            print(cnt)

    ouf = open(out_path, "w")
    print_res(law, crit, time, ouf)


def work(from_id, to_id):
    global global_crit,global_time,global_law
    for a in range(0, get_num_classes("crit")):
        global_crit.append(0)
    for a in range(0, get_num_classes("time")):
        global_time.append(0)
    for a in range(0, get_num_classes("law1")):
        global_law.append(0)
    for a in range(int(from_id), int(to_id)):
        print(str(a) + " begin to work")
        draw_out(os.path.join(in_path, str(a)), os.path.join(out_path, str(a)))
        print(str(a) + " work done")


if __name__ == "__main__":
    work(0, 20)

    ouf = open("result/count_data/total.txt", "w")

    print_res(global_law, global_crit, global_time, ouf)
