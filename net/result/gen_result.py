import json

f = open("count_data/total.txt", "r")
nowb = ""

for line in f:
    line = line[:-1]
    if line == "law":
        nowb = "law"
        ouf = open("law_result.txt", "w")
    elif line == "crit":
        nowb = "crit"
        ouf = open("crit_result.txt", "w")
    elif line == "time":
        break
    else:
        if line == "":
            continue
        if nowb == "crit":
            data = line.split(" ")
            a = data[0]
            b = int(data[1])
            print(a, b, file=ouf)
        elif nowb == "law":
            data = line.replace("(", "").replace(",", "").replace(")", "").split(" ")
            a = int(data[0])
            b = int(data[1])
            c = int(data[2])
            d = int(data[3])
            print(a, b, c, d, file=ouf)
        else:
            pass
