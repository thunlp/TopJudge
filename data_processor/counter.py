# coding: UTF-8

import os
import json
import re

# in_path = r"D:\work\law_pre\test\in"
# out_path = r"D:\work\law_pre\test\out"
in_path = r"/disk/mysql/law_data/final_data"
out_path = r"/disk/mysql/law_data/count_data"
mid_text = u"  _(:з」∠)_  "
title_list = ["docId", "caseNumber", "caseName", "spcx", "court", "time", "caseType", "bgkly", "yuanwen", "document",
              "cause", "docType", "keyword", "lawyer", "punishment", "result", "judge"]

accusation_file = r"/home/zhx/law_pre/data_processor/accusation_list2.txt"
accusation_f = open(accusation_file, "r", encoding='utf8')
accusation_list = json.loads(accusation_f.readline())
# accusation_list = []
# for line in accusation_f:
#    accusation_list.append(line[:-1])

num_file = 1
num_process = 1

total_cnt = 0

youqi_list = {}
juyi_list = {}
guanzhi_list = {}
wuqi_cnt = 0
sixing_cnt = 0


def gen_youqi(data):
    ans = -1
    for x in data:
        if x > ans:
            ans = x
    return ans


def gen_juyi(data):
    ans = -1
    for x in data:
        if x > ans:
            ans = x
    return ans


def gen_guanzhi(data):
    ans = -1
    for x in data:
        if x > ans:
            ans = x
    return ans


def analyze_time(data):
    if data == {}:
        return
    global youqi_list
    global juyi_list
    global guanzhi_list
    global wuqi_cnt
    global sixing_cnt

    if data["sixing"]:
        sixing_cnt += 1
        return

    if data["wuqi"]:
        wuqi_cnt += 1
        return

    if len(data["youqi"]) != 0:
        x = gen_youqi(data["youqi"])
        if not (x in youqi_list):
            youqi_list[x] = 0
        youqi_list[x] += 1
        return

    if len(data["juyi"]) != 0:
        x = gen_juyi(data["juyi"])
        if not (x in juyi_list):
            juyi_list[x] = 0
            juyi_list[x] += 1
        return

    if len(data["guanzhi"]) != 0:
        x = gen_guanzhi(data["guanzhi"])
        if not (x in guanzhi_list):
            guanzhi_list[x] = 0
        guanzhi_list[x] += 1
        return


money_list = {}


def gen_money(data):
    ans = -1
    for x in data:
        if x > ans:
            ans = x
    return ans


def analyze_money(data):
    if len(data) == 0:
        return
    global money_list
    x = gen_money(data)
    if not (x in money_list):
        money_list[x] = 0
    money_list[x] += 1


law_list = {}
law_list["only_name"] = {}
law_list["name_tiao"] = {}
law_list["name_tiao_kuan"] = {}


def analyze_law(data):
    if len(data) == 0:
        return
    global law_list
    for r in data:
        x = r["law_name"]
        y = r["tiao_num"]
        z = r["kuan_num"]
        if not (x in law_list["only_name"]):
            law_list["only_name"][x] = 0
            law_list["name_tiao"][x] = {}
            law_list["name_tiao_kuan"][x] = {}

        law_list["only_name"][x] += 1

        if not (str(y) in law_list["name_tiao"][x]):
            law_list["name_tiao"][x][str(y)] = 0
        law_list["name_tiao"][x][str(y)] += 1

        if not (str((y, z)) in law_list["name_tiao_kuan"][x]):
            law_list["name_tiao_kuan"][x][str((y, z))] = 0
        law_list["name_tiao_kuan"][x][str((y, z))] += 1


crit_list = []
for a in range(0, len(accusation_list)):
    crit_list.append(0)


def analyze_crit(data):
    if len(data) == 0:
        return
    global crit_list
    for x in data:
        for a in range(0, len(accusation_list)):
            if x.replace("[", "").replace("]", "") == accusation_list[a].replace("[", "").replace("]", ""):
                crit_list[a] += 1
                break


def count(data):
    global total_cnt
    total_cnt += 1

    analyze_time(data["term_of_imprisonment"])
    analyze_money(data["punish_of_money"])
    analyze_law(data["name_of_law"])
    analyze_crit(data["name_of_accusation"])


def draw_out(in_path, out_path):
    print(in_path)
    inf = open(in_path, "r")
    ouf = open(out_path, "w")

    cnt = 0
    for line in inf:
        data = json.loads(line)
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
    # import multiprocessing

    # process_pool = []

    # for a in range(0, num_process):
    #    process_pool.append(
    #        multiprocessing.Process(target=work, args=(a * num_file / num_process, (a + 1) * num_file / num_process)))

    # for a in process_pool:
    #    a.start()

    # for a in process_pool:
    #    a.join()

    ouf = open("result/result.txt", "w")
    data = {}
    data["total"] = total_cnt
    data["youqi"] = youqi_list
    data["wuqi"] = wuqi_cnt
    data["juyi"] = juyi_list
    data["guanzhi"] = guanzhi_list
    data["sixing"] = sixing_cnt

    data["law"] = law_list
    data["money"] = money_list
    data["crit"] = crit_list
    print(json.dumps(data), file=ouf)
