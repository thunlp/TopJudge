# coding: UTF-8

import os
import json
import re
from net.parser import ConfigParser

from net.data_formatter import get_time_id, check_sentence
from net.loader import get_name

in_path = r"/data/zhx/law/data/cail"
out_path = r"/disk/mysql/law_data/count_data"

num_file = 20
num_process = 1

total_cnt = 0

crit = {}
law = {}
term = {}

config = ConfigParser("/home/zhx/law_pre/config/default_config.config")


def analyze_law(data):
    for x, y, z in data:
        if x < 102 or x > 452:
            continue
        if not ((x, y) in law.keys()):
            law[(x, y)] = 0
        law[(x, y)] += 1


def analyze_crit(data):
    for x in data:
        if not (x in crit.keys()):
            crit[x] = 0
        crit[x] += 1


def analyze_time(data):
    idx = get_time_id(data, None)
    name = get_name("time", idx)
    if not (name in term.keys()):
        term[name] = 0
    term[name] += 1


def count(data):
    global total_cnt
    total_cnt += 1

    analyze_law(data["law"])
    analyze_crit(data["crit"])
    analyze_time(data["time"])


def check(data):
    if len(data["meta"]["crit"]) != 1:
        return False
    cnt = 0

    arr = []
    for x, y, z in data["meta"]["law"]:
        if x < 102 or x > 452:
            continue
        arr.append((x, y))

    arr = list(set(arr))
    arr.sort()

    return len(arr) == 1


def draw_out(in_path, out_path):
    print(in_path)
    inf = open(in_path, "r")

    cnt = 0
    for line in inf:
        data = json.loads(line)
        if not (check(data)):
            continue
        if not (check_sentence(data["content"],config)):
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
    print(total_cnt)

    f = open(os.path.join(in_path, "crit.txt"), "w")
    for x in crit.keys():
        print(x, crit[x], file=f)
    f.close()

    f = open(os.path.join(in_path, "time.txt"), "w")
    for x in term.keys():
        print(x, term[x], file=f)
    f.close()

    f = open(os.path.join(in_path, "law.txt"), "w")
    for x, y in law.keys():
        print(x, y, law[(x, y)], file=f)
    f.close()

    f = open(os.path.join(in_path, "total.txt"), "w")
    print(total_cnt, file=f)
    f.close()
