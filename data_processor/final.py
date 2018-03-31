# coding: UTF-8

import os
import json
import thulac
import re

cutter = thulac.thulac(model_path=r"/home/zhx/models", seg_only=True, filt=False)

in_path = "/disk/mysql/law_data/classified_data/刑事判决书"
out_path = "/disk/mysql/law_data/final_data2/"
mid_text = u"\t"
title_list = ["docId", "caseNumber", "caseName", "spcx", "court", "time", "caseType", "bgkly", "yuanwen", "document",
              "cause", "docType", "keyword", "lawyer", "punishment", "result", "judge"]

min_length = 32
max_length = 1024

accusation_file = "/home/zhx/law_pre/data_processor/accusation_list2.txt"
f = open(accusation_file, "r")
accusation_list = json.loads(f.readline())
for a in range(0, len(accusation_list)):
    accusation_list[a] = accusation_list[a].replace('[', '').replace(']', '')
f.close()

num_process = 1
num_file = 20


def cut(s):
    data = cutter.cut(s)
    result = ""
    first = True
    for x, y in data:
        if x == " ":
            continue
        if first:
            first = False
        else:
            result = result + mid_text
        result = result + x
    return result


def format_string(s):
    return s.replace("b", "").replace("\t", " ").replace("t", "")


def generate_fact(data):
    if "AJJBQK" in data["document"]:
        s = format_string(data["document"]["AJJBQK"])
        regex_list = [
            (r"(经审理查明|公诉机关指控|检察院指控)([，：,:]?)([\s\S]*)([，。,]?)(足以认定|就上述指控|上述事实)", 2),
            (r"(经审理查明|公诉机关指控|检察院指控)([，：,:]?)([\s\S]*)$", 2),
            (r"^([\s\S]*)([，。,]?)(足以认定|就上述指控|上述事实)", 0)
        ]

        fact = None

        for reg, num in regex_list:
            regex = re.compile(reg)
            result = re.findall(regex, s)
            if len(result) > 0:
                fact = result[0][num]
                break
        if not (fact is None):
            return fact

    if "SSJL" in data["document"]:
        s = format_string(data["document"]["SSJL"])
        regex_list = [
            (r"(经审理查明|公诉机关指控|检察院指控)([，：,:]?)([\s\S]*)([，。,]?)(足以认定|就上述指控|上述事实])", 2),
            (r"(经审理查明|公诉机关指控|检察院指控)([，：,:]?)([\s\S]*)$", 2),
            (r"^([\s\S]*)([，。,]?)(足以认定|就上述指控|上述事实)", 0)
        ]

        fact = None

        for reg, num in regex_list:
            regex = re.compile(reg)
            result = re.findall(regex, s)
            if len(result) > 0:
                fact = result[0][num]
                break
        if not (fact is None):
            return fact

    if "content" in data["document"]:
        s = format_string(data["document"]["content"])
        regex_list = [
            (r"(经审理查明|公诉机关指控|检察院指控)([，：,:]?)([\s\S]*)([，。,]?)(足以认定|就上述指控|上述事实)", 2),
            (r"(经审理查明|公诉机关指控|检察院指控)([，：,:]?)([\s\S])*事实一致", 2)
        ]

        fact = None

        for reg, num in regex_list:
            regex = re.compile(reg)
            result = re.findall(regex, s)
            if len(result) > 0:
                fact = result[0][num]
                break
        if not (fact is None):
            return fact

        print(s)


def draw_out(in_path, out_path):
    print(in_path)
    inf = open(in_path, "r")
    ouf = open(out_path, "w")

    cnt = 0
    cx = 0
    for line in inf:
        try:
            data = json.loads(line)
            fact = generate_fact(data)
            # print(fact)

            cnt += 1
            if cnt % 50000 == 0:
                gg
                print(in_path, cnt, cx)
                # break

        except Exception as e:
            gg  # print(e)
            # gg


def work(from_id, to_id):
    for a in range(int(from_id), int(to_id)):
        print(str(a) + " begin to work")
        draw_out(os.path.join(in_path, str(a)), os.path.join(out_path, str(a)))
        print(str(a) + " work done")


if __name__ == "__main__":
    import multiprocessing

    process_pool = []

    for a in range(0, num_process):
        process_pool.append(
            multiprocessing.Process(target=work, args=(a * num_file / num_process, (a + 1) * num_file / num_process)))

    for a in process_pool:
        a.start()

    for a in process_pool:
        a.join()
