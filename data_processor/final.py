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

num_list = {
    u"〇": 0,
    u"\uff2f": 0,
    u"\u3007": 0,
    u"\u25cb": 0,
    u"\uff10": 0,
    u"\u039f": 0,
    u'零': 0,
    "O": 0,
    "0": 0,
    u"一": 1,
    u"元": 1,
    u"1": 1,
    u"二": 2,
    u"2": 2,
    u"两": 2,
    u'三': 3,
    u'3': 3,
    u'四': 4,
    u'4': 4,
    u'五': 5,
    u'5': 5,
    u'六': 6,
    u'6': 6,
    u'七': 7,
    u'7': 7,
    u'八': 8,
    u'8': 8,
    u'九': 9,
    u'9': 9,
    u'十': 10,
    u'百': 100,
    u'千': 1000,
    u'万': 10000
}

num_str = ""

for x in num_list:
    num_str = num_str + x


def parse_date_with_year_and_month_begin_from(s, begin, delta):
    # erf = open("error.log", "a")
    pos = begin + delta
    num1 = 0
    while s[pos] in num_list:
        if s[pos] == u"十":
            if num1 == 0:
                num1 = 1
            num1 *= 10
        elif s[pos] == u"百" or s[pos] == u"千" or s[pos] == u"万":
            # print("0 " + s[begin - 10:pos + 20], file=erf)
            return None
        else:
            num1 = num1 + num_list[s[pos]]

        pos += 1

    num = 0
    if s[pos] == u"年":
        num2 = 0
        pos += 1
        if s[pos] == u"又":
            pos += 1
        while s[pos] in num_list:
            if s[pos] == u"十":
                if num2 == 0:
                    num2 = 1
                num2 *= 10
            elif s[pos] == u"百" or s[pos] == u"千" or s[pos] == u"万":
                # print("1 " + s[begin - 10:pos + 20], file=erf)
                return None
            else:
                num2 = num2 + num_list[s[pos]]

            pos += 1
        if s[pos] == u"个":
            pos += 1
        if num2 != 0 and s[pos] != u"月":
            # print("2 " + s[begin - 10:pos + 20], file=erf)
            return None
        num = num1 * 12 + num2
    else:
        if s[pos] == u"个":
            pos += 1
        if s[pos] != u"月":
            # print("3 " + s[begin - 10:pos + 20], file=erf)
            return None
        else:
            num = num1

    pos += 1
    # print(num,s[x.start():pos])
    return num


def parse_term_of_imprisonment(data):
    result = {}
    if "PJJG" in data["document"]:
        s = data["document"]["PJJG"].replace('b', '')

        # 有期徒刑
        youqi_arr = []
        pattern = re.compile(u"有期徒刑")
        for x in pattern.finditer(s):
            pos = x.start()
            data = parse_date_with_year_and_month_begin_from(s, pos, len(u"有期徒刑"))
            if not (data is None):
                youqi_arr.append(data)

        # 拘役
        juyi_arr = []
        pattern = re.compile(u"拘役")
        for x in pattern.finditer(s):
            pos = x.start()
            data = parse_date_with_year_and_month_begin_from(s, pos, len(u"拘役"))
            if not (data is None):
                juyi_arr.append(data)

        # 管制
        guanzhi_arr = []
        pattern = re.compile(u"管制")
        for x in pattern.finditer(s):
            pos = x.start()
            data = parse_date_with_year_and_month_begin_from(s, pos, len(u"管制"))
            if not (data is None):
                guanzhi_arr.append(data)

        # 无期
        forever = False
        if s.count("无期徒刑") != 0:
            forever = True

        # 死刑
        dead = False
        if s.count("死刑") != 0:
            dead = True

        result["youqi"] = youqi_arr
        result["juyi"] = juyi_arr
        result["guanzhi"] = guanzhi_arr
        result["wuqi"] = forever
        result["sixing"] = dead

    return result


def dfs_search(s, x, p, y):
    if p >= len(x):
        return s.count(y) != 0
    if x[p] == "[":
        pp = p
        while x[pp] != "]":
            pp += 1
        subs = x[p + 1:pp].split(u"、")
        for z in subs:
            if dfs_search(s, x, pp + 1, y + z):
                return True
        if dfs_search(s, x, pp + 1, y + x[p + 1:pp]):
            return True
        else:
            return False
    else:
        return dfs_search(s, x, p + 1, y + x[p])


def check(x, s):
    return dfs_search(s, x, 0, "")


def parse_name_of_accusation(data):
    if "PJJG" in data["document"]:
        s = data["document"]["PJJG"]
        result = []
        for x in accusation_list:
            if check(x, s):
                result.append(x.replace("[", "").replace("]", ""))
        # print(result)
        # if len(result) == 0:
        #    print(s)
        return result
    else:
        return []


key_word_list = [u"第", u"条", u"款", u"、", u"，", u"（", u"）", u"之"]


def get_number_from_string(s):
    for x in s:
        if not (x in num_list):
            print(s)
            gg

    value = 0
    try:
        value = int(s)
    except ValueError:
        nowbase = 1
        addnew = True
        for a in range(len(s) - 1, -1, -1):
            if s[a] == u'十':
                if nowbase >= 10000:
                    nowbase = 100000
                else:
                    nowbase = 10
                addnew = False
            elif s[a] == u'百':
                if nowbase >= 10000:
                    nowbase = 1000000
                else:
                    nowbase = 100
                addnew = False
            elif s[a] == u'千':
                if nowbase >= 10000:
                    nowbase = 10000000
                else:
                    nowbase = 1000
                addnew = False
            elif s[a] == u'万':
                nowbase = 10000
                addnew = False
            else:
                value = value + nowbase * num_list[s[a]]
                nowbase = nowbase * 10
                addnew = True

        if not (addnew):
            value += nowbase

    return value


def get_one_reason(content, rex):
    pos = rex.start()
    law_name = rex.group(1)
    nows = rex.group().replace(u"（", u"").replace(u"）", u"")
    # print(nows)

    result = []

    p = 0
    while nows[p] != u"》":
        p += 1
    while nows[p] != u"第":
        p += 1

    tiao_num = 0
    kuan_num = 0
    add_kuan = True
    zhiyi = 0
    while p < len(nows):
        nowp = p + 1
        while not (nows[nowp] in key_word_list):
            nowp += 1
        num = get_number_from_string(nows[p + 1:nowp])
        if nows[nowp] != u"款":
            if not (add_kuan):
                result.append({"law_name": law_name, "tiao_num": tiao_num, "kuan_num": 0, "zhiyi": zhiyi})
            tiao_num = num
            add_kuan = False
            if len(nows) > nowp + 2 and nows[nowp + 1] == u"之" and nows[nowp + 2] in num_list:
                if num_list[nows[nowp + 2]] != 1:
                    print(nows)
                    # gg
                zhiyi = num_list[nows[nowp + 2]]
            else:
                zhiyi = 0
        else:
            kuan_num = num
            result.append({"law_name": law_name, "tiao_num": tiao_num, "kuan_num": kuan_num, "zhiyi": zhiyi})
            add_kuan = True

        p = nowp

        while p < len(nows) and nows[p] != u'第':
            p += 1

    if not (add_kuan):
        result.append({"law_name": law_name, "tiao_num": tiao_num, "kuan_num": 0, "zhiyi": zhiyi})
    # print(result)
    # if zhiyi == 1:
    #    gg

    # print nows
    # for x in result:
    #    print x["law_name"], x["tiao_num"], x["kuan_num"]
    # print

    return result


def sort_reason(l):
    result_list = []

    law_list = {}

    for x in l:
        z = x
        if not (z["law_name"] in law_list):
            law_list[z["law_name"]] = set()
        law_list[z["law_name"]].add((z["tiao_num"], z["kuan_num"], z["zhiyi"]))

    for x in law_list:
        gg = []
        for (y, z, r) in law_list[x]:
            gg.append((y, z, r))
        gg = list(set(gg))
        gg.sort()
        for (y, z, r) in gg:
            result_list.append({"law_name": x, "tiao_num": y, "kuan_num": z, "zhiyi": r})

    return result_list


def parse_name_of_law(data):
    if not ("content" in data["document"]):
        return []

    key_word_str = num_str
    for x in key_word_list:
        key_word_str = key_word_str + x
    rex = re.compile(u"《(中华人民共和国刑法)》第[" + key_word_str + u"]*[条款]")
    result = rex.finditer(data["document"]["content"])

    result_list = []

    law_list = {}

    for x in result:
        y = get_one_reason(data["document"]["content"], x)
        for z in y:
            if not (z["law_name"] in law_list):
                law_list[z["law_name"]] = set()
            law_list[z["law_name"]].add((z["tiao_num"], z["kuan_num"], z["zhiyi"]))

    for x in law_list:
        for (y, z, r) in law_list[x]:
            result_list.append({"law_name": x, "tiao_num": y, "kuan_num": z, "zhiyi": r})

    return sort_reason(result_list)


def parse_money(data):
    if not ("PJJG" in data["document"]):
        return []

    result_list = []

    rex = re.compile(u"人民币([" + num_str + "]*)元")
    result = rex.finditer(data["document"]["PJJG"])

    for x in result:
        datax = get_number_from_string(x.group(1))
        result_list.append(datax)
        # print(x.group(1), datax)

    # print(result_list)

    return result_list


def parse_criminals(data):
    se = set()
    if "DSRXX" in data["document"]:
        s = format_string(data["document"]["DSRXX"])
        regex = re.compile(r"被告人((自报|：|,)?)(\S{2,4})[，。、,.（\(]")
        se = set()

        for result in re.finditer(regex, s):
            se.add(result.group(3))

    if not("DSRXX" in data["document"]) or len(se) == 0:
        s = format_string(data["document"]["content"])
        regex = re.compile(r"被告人((自报|：|，)?)(\S{2,4})[，。、,.（\(]")
        se = set()

        for result in re.finditer(regex, s):
            se.add(result.group(3))

    if len(se) == 0:
        print(se)
        print(data["document"]["content"])
    return se



def parse(data):
    result = {}
    # print(data["document"]["PJJG"])

    result["criminals"] = parse_criminals(data)

    return result
    result["term_of_imprisonment"] = parse_term_of_imprisonment(data)
    result["name_of_accusation"] = parse_name_of_accusation(data)
    result["name_of_law"] = parse_name_of_law(data)
    result["punish_of_money"] = parse_money(data)

    return result


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
            (r"(经审理查明|公诉机关指控|检察院指控|起诉书指控)([，：,:]?)([\s\S]*)([，。,]?)(足以认定|就上述指控|上述事实)", 2),
            (r"(经审理查明|公诉机关指控|检察院指控|起诉书指控)([，：,:]?)([\s\S]*)$", 2),
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
            (r"(经审理查明|公诉机关指控|检察院指控|起诉书指控)([，：,:]?)([\s\S]*)([，。,]?)(足以认定|就上述指控|上述事实)", 2),
            (r"(经审理查明|公诉机关指控|检察院指控|起诉书指控)([，：,:]?)([\s\S]*)$", 2),
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
            (r"(经审理查明|公诉机关指控|检察院指控|起诉书指控)([，：,:]?)([\s\S]*)([，。,]?)(足以认定|就上述指控|上述事实)", 2),
            (r"(经审理查明|公诉机关指控|检察院指控|起诉书指控)([，：,:]?)([\s\S])*事实一致", 2),
            (r"指控([，：,:])([\s\S]*)。本院认为", 1),
            (r"(经审理查明|公诉机关指控|检察院指控|起诉书指控)([，：,:]?)([\s\S])*《", 2),
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
    return None

    print(data["document"]["Title"])
    if "AJJBQK" in data["document"]:
        print(data["document"]["AJJBQK"])
    else:
        print("no AJJBQK")
    if "SSJL" in data["document"]:
        print(data["document"]["SSJL"])
    else:
        print("no SSJL")
    if "content" in data["document"]:
        print(data["document"]["content"])
    else:
        print("no content")
    print("")


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
            if fact is None:
                continue
            # print(fact)
            data["meta"] = parse(data)
            """if not ("youqi" in data["meta"]["term_of_imprisonment"]) or len(
                    data["meta"]["term_of_imprisonment"]["youqi"]) <= 1:
                continue
            print(data["document"]["Title"])
            print("content", data["document"]["content"])
            print("fact", fact)
            if "PJJG" in data["document"]:
                print("result", data["document"]["PJJG"])
            else:
                print("result no result")
            print("meta", data["meta"])
            print("")"""

            cnt += 1
            if cnt % 5000 == 0:
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
