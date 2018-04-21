min_frequency = 100

# 10 214 474

accusation_list = []
accusation_dict = {}
f = open("net/result/crit_result.txt", "r")
for line in f:
    data = line[:-1].split(" ")
    name = data[0]
    num = int(data[1])
    if num > min_frequency:
        accusation_list.append(name)
        accusation_dict[name] = len(accusation_list) - 1

law_list = []
law_dict = {}
f = open("net/result/law_result.txt", "r")
for line in f:
    data = line[:-1].split(" ")
    name = (int(data[0]), int(data[1]), int(data[2]))
    num = int(data[3])
    if num > min_frequency:
        law_list.append(name)
        law_dict[name] = len(law_list) - 1

law_list_tiao = []
law_dict_tiao = {}
f = open("net/result/law_result_tiao.txt", "r")
for line in f:
    data = line[:-1].split(" ")
    name = (int(data[0]), int(data[1]))
    num = int(data[2])
    if num > min_frequency:
        law_list_tiao.append(name)
        law_dict_tiao[name] = len(law_list_tiao) - 1


def get_num_classes(s):
    if s == "crit":
        return len(accusation_list)
    if s == "law":
        return len(law_list)
    if s == "law1":
        return len(law_list_tiao)
    if s == "time":
        return 11
    gg


def get_name(s, num):
    if s == "crit":
        return accusation_list[num]
    if s == "law":
        return law_list[num]
    if s == "law1":
        return law_list_tiao[num]
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


print(len(accusation_list))
# print(len(law_list))
print(len(law_list_tiao))
