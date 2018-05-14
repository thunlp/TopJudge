import os

accusation_list = []
accusation_dict = {}
law_list = []
law_dict = {}


def init(config):
    min_frequency = config.getint("data", "min_frequency")
    data_path = os.path.join(config.get("data", "data_path"), config.get("data", "dataset"))
    f = open(os.path.join(data_path, "crit.txt"), "r")
    cnt1 = 0
    for line in f:
        data = line[:-1].split(" ")
        name = data[0]
        num = int(data[1])
        if num > min_frequency:
            cnt1 += num
            accusation_list.append(name)
            accusation_dict[name] = len(accusation_list) - 1

    cnt2 = 0
    f = open(os.path.join(data_path, "law.txt"), "r")
    for line in f:
        data = line[:-1].split(" ")
        name = (int(data[0]), int(data[1]))
        num = int(data[2])
        if num > min_frequency:
            cnt2 += num
            law_list.append(name)
            law_dict[name] = len(law_list) - 1

    print(len(accusation_list), cnt1)
    print(len(law_list), cnt2)


def get_num_classes(s):
    if s == "crit":
        return len(accusation_list)
    if s == "law":
        return len(law_list)
    if s == "time":
        return 11
    gg


def get_name(s, num):
    if s == "crit":
        return accusation_list[num]
    if s == "law":
        return law_list[num]
    if s == "time":
        map_list = {
            0: "死刑或无期",
            1: "十年以上",
            2: "七到十年",
            3: "五到七年",
            4: "三到五年",
            5: "二到三年",
            6: "一到二年",
            7: "九到十二个月",
            8: "六到九个月",
            9: "零到六个月",
            10: "没事"
        }

        return map_list[num]

    gg
