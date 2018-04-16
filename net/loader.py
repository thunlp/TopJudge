min_frequency = 10

accusation_list = []
accusation_dict = {}
f = open("result/crit_result.txt", "r")
for line in f:
    data = line[:-1].split(" ")
    name = data[0]
    num = int(data[1])
    if num > min_frequency:
        accusation_list.append(name)
        accusation_dict[name] = len(accusation_list) - 1

law_list = []
law_dict = {}
f = open("result/law_result1.txt", "r")
for line in f:
    data = line[:-1].split(" ")
    name = (int(data[0]), int(data[1]), int(data[2]))
    num = int(data[3])
    if num > min_frequency:
        law_list.append(name)
        law_dict[name] = len(law_list) - 1