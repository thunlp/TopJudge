file_path = r"C:\work\law_pre\data_processor\result\result.txt"
crit_path = r"C:\work\law_pre\data_processor\accusation_list2.txt"
f = open(file_path, "r")

import json

data = json.loads(f.readline())
f.close()

f = open(crit_path, "r")
crit_list = json.loads(f.readline())
f.close()

print("total:", data["total"])


def analyze_lawname():
    f = open("result/name_result.txt", "w")
    cnt = 0
    for x in data["law"]["name_tiao_kuan"].keys():
        if x == u"中华人民共和国刑法":
            arr = []
            nowd = data["law"]["name_tiao_kuan"][x]
            for rs in nowd.keys():
                r = rs.replace("(", "").replace(")", "").split(",")
                (x, y) = int(r[0]), int(r[1])
                arr.append((x, y, nowd[rs]))
            arr.sort()
            for (x, y, z) in arr:
                print(x, y, z, file=f)
                cnt += z
            break
    print("name total:", cnt)

    pass


def analyze_critname():
    cnt = 0
    f = open("result/crit_result.txt", "w")
    for a in range(0, len(crit_list)):
        print(crit_list[a], data["crit"][a], file=f)
        cnt += data["crit"][a]
    f.close()
    print("crit total:", cnt)
    pass


def analyze_time():
    f = open("result/time_result.txt", "w")
    youqi_time = [0, 2, 3, 5, 7, 10, 15, 25]
    res = {}
    res["[0,0]"] = 0
    res["(0,0.6]"] = 0
    res["(0.6,0.9]"] = 0
    res["(0.9,1]"] = 0
    res["(1,2]"] = 0
    res["(2,3]"] = 0
    res["(3,5]"] = 0
    res["(5,7]"] = 0
    res["(7,10]"] = 0
    res["(10,15]"] = 0
    res["(15,25]"] = 0
    res["(25,∞]"] = 0
    res["gg"] = data["sixing"]
    for x in data["youqi"].keys():
        y = int(x) / 12
        if y == 0:
            res["[0,0]"] += data["youqi"][x]
        elif y <= 1:
            y = int(x)
            if y <= 6:
                res["(0,0.6]"] += data["youqi"][x]
            elif y <= 9:
                res["(0.6,0.9]"] += data["youqi"][x]
            else:
                res["(0.9,1]"] += data["youqi"][x]
        elif y <= 2:
            res["(1,2]"] += data["youqi"][x]
        elif y <= 3:
            res["(2,3]"] += data["youqi"][x]
        elif y <= 5:
            res["(3,5]"] += data["youqi"][x]
        elif y <= 7:
            res["(5,7]"] += data["youqi"][x]
        elif y <= 10:
            res["(7,10]"] += data["youqi"][x]
        elif y <= 15:
            res["(10,15]"] += data["youqi"][x]
        elif y <= 25:
            res["(15,25]"] += data["youqi"][x]
        else:
            res["(25,∞]"] += data["youqi"][x]

    res["(25,∞]"] += data["wuqi"]

    for x in data["guanzhi"].keys():
        y = int(x) / 12
        if y == 0:
            res["[0,0]"] += data["guanzhi"][x]
        elif y <= 1:
            y = int(x)
            if y <= 6:
                res["(0,0.6]"] += data["guanzhi"][x]
            elif y <= 9:
                res["(0.6,0.9]"] += data["guanzhi"][x]
            else:
                res["(0.9,1]"] += data["guanzhi"][x]
        elif y <= 2:
            res["(1,2]"] += data["guanzhi"][x]
        elif y <= 2:
            res["(0,2]"] += data["guanzhi"][x]
        elif y <= 3:
            res["(2,3]"] += data["guanzhi"][x]
        else:
            res["(3,5]"] += data["guanzhi"][x]

    for x in data["juyi"].keys():
        y = int(x) / 12
        if y == 0:
            res["[0,0]"] += data["juyi"][x]
        elif y <= 1:
            y = int(x)
            if y <= 6:
                res["(0,0.6]"] += data["juyi"][x]
            elif y <= 9:
                res["(0.6,0.9]"] += data["juyi"][x]
            else:
                res["(0.9,1]"] += data["juyi"][x]
        elif y <= 2:
            res["(1,2]"] += data["juyi"][x]
        elif y <= 2:
            res["(0,2]"] += data["juyi"][x]

    cnt = 0
    for x in res.keys():
        print(x, res[x], file=f)
        cnt += res[x]

    print("time total:", cnt)

    pass


def analyze_money():
    pass


for x in data.keys():
    print(x)

analyze_time()
analyze_lawname()
analyze_critname()
analyze_money()
