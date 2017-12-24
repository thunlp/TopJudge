# coding: UTF-8

import os
import json
import thulac

cutter = thulac.thulac(model_path=r"C:\work\thulac\Models_v1_v2\models", seg_only=True, filt=True)

in_path = "/disk/mysql/law_data/critical_data/"
out_path = "/disk/mysql/law_data/final_data/"
mid_text = u" qwq "
title_list = ["docId", "caseNumber", "caseName", "spcx", "court", "time", "caseType", "bgkly", "yuanwen", "document",
              "cause", "docType", "keyword", "lawyer", "punishment", "result", "judge"]

min_length = 25
max_length = 1000

num_process = 1
num_file = 1


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
        return None, False
    res = data
    res["youqi"] = list(set(res[youqi]))
    res["youqi"].sort()
    res["guanzhi"] = list(set(res[youqi]))
    res["guanzhi"].sort()
    res["juyi"] = list(set(res[youqi]))
    res["juyi"].sort()

    return res, True


money_list = {}


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
        if x == u"中华人民共和国刑法":
            res.append((y, z))

    res = list(set(res))
    res.sort()

    return res, len(res) == 0


def analyze_crit(data):
    if len(data) == 0:
        return None, False
    data = list(set(data))
    data.sort()
    return data, True


def analyze_meta(data):
    res = {}
    res["time"], able1 = analyze_time(data["term_of_imprisonment"])
    res["law"], able2 = analyze_law(data["name_of_law"])
    res["crit"], able3 = analyze_crit(data["name_of_accusation"])

    return res, able1 and able2 and able3


def cut(s):
    data = cutter.cut(s)
    result = ""
    first = True
    for x, y in data:
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

                s = data["documnt"]["AJJBQK"].replace("b", "")
                l = len(data["document"]["AJJBQK"])
                if l >= min_length and l <= max_length:
                    res["content"] = cut(s)

                res["meta"], able = analyze_meta(data["meta_info"])
                if not (able):
                    continue

                print(json.dumps(res), file=ouf)

            cnt += 1
            if cnt == 50:
                print(in_path, cnt, cx)
                break

        except Exception as e:
            print(e)
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
