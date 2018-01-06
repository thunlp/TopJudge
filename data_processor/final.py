# coding: UTF-8

import os
import json
import thulac

cutter = thulac.thulac(model_path=r"/home/zhx/models", seg_only=True, filt=False)

in_path = "/disk/mysql/law_data/critical_data/"
out_path = "/disk/mysql/law_data/final_data/"
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

num_process = 4
num_file = 20


def analyze_time(data):
    if data == {}:
        return None, False
    res = data
    res["youqi"] = list(set(res["youqi"]))
    res["youqi"].sort()
    res["guanzhi"] = list(set(res["guanzhi"]))
    res["guanzhi"].sort()
    res["juyi"] = list(set(res["juyi"]))
    res["juyi"].sort()

    return res, True


def analyze_money(data):
    pass


def analyze_law(data):
    if len(data) == 0:
        return None, False
    cnt = 0
    res = []
    for r in data:
        x = r["law_name"]
        y = r["tiao_num"]
        z = r["kuan_num"]
        f = r["zhiyi"]
        if x == u"中华人民共和国刑法":
            res.append((y, z, f))

    res = list(set(res))
    res.sort()

    return res, len(res) != 0


def analyze_crit(data):
    if len(data) == 0:
        return None, False
    for a in range(0, len(data)):
        find = False
        for b in range(0, len(accusation_list)):
            if data[a] == accusation_list[b]:
                data[a] = b
                find = True
                break
        if not (find):
            print(accusation_list[38])
            print(data[a])
            gg
    data = list(set(data))
    # print(data)
    data.sort()
    return data, True


def analyze_meta(data):
    res = {}
    res["time"], able1 = analyze_time(data["term_of_imprisonment"])
    res["law"], able2 = analyze_law(data["name_of_law"])
    res["crit"], able3 = analyze_crit(data["name_of_accusation"])
    # print(able1,able2,able3)

    return res, able1 and able2 and able3


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


def draw_out(in_path, out_path):
    print(in_path)
    inf = open(in_path, "r")
    ouf = open(out_path, "w")

    cnt = 0
    cx = 0
    for line in inf:
        try:
            data = json.loads(line)
            if "AJJBQK" in data["document"]:
                res = {}

                # filter_list = [65292, 12290, 65311, 65281, 65306, 65307, 8216, 8217, 8220, 8221, 12304, 12305,
                #               12289, 12298, 12299, 126, 183, 64, 124, 35, 65509, 37, 8230, 38, 42, 65288,
                #               65289, 8212, 45, 43, 61, 44, 46, 60, 62, 63, 47, 33, 59, 58, 39, 34, 123, 125,
                #               91, 93, 92, 124, 35, 36, 37, 94, 38, 42, 40, 41, 95, 45, 43, 61, 9700, 9734, 9733]
                s = data["document"]["AJJBQK"].replace("b", "").replace("\t", "")
                # for x in filter_list:
                #    s = s.replace(chr(x), ' ')

                s = cut(s)

                l = len(s.split(mid_text))
                # print(l)
                if l >= min_length and l <= max_length:
                    res["content"] = s
                else:
                    # print(s)
                    continue

                res["meta"], able = analyze_meta(data["meta_info"])
                if not (able):
                    continue

                cx += 1
                print(json.dumps(res), file=ouf)

            cnt += 1
            if cnt % 50000 == 0:
                print(in_path, cnt, cx)
                # break

        except Exception as e:
            pass  # print(e)
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
