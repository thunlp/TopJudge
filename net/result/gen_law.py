law_dict = {}
f = open("law_result.txt", "r")
for line in f:
    data = line[:-1].split(" ")
    name = (int(data[0]), int(data[1]))
    num = int(data[3])

    if not (name in law_dict):
        law_dict[name] = 0

    law_dict[name] += num

f = open("law_result_tiao.txt", "w")

for x in law_dict.keys():
    print(x[0], x[1], law_dict[x], file=f)

f.close()
