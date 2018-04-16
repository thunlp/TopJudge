min_frequency = 10

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


def get_num_classes(s):
    if s == "crit":
        return len(accusation_list)
    if s == "law":
        return len(law_list)
    if s == "time":
        return 11
    gg
