import os
import json
import torch
import random
import numpy as np

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


def check_crit(data):
    for x in data:
        if not (x in accusation_dict.keys()):
            return False
    return True


def check_law(data):
    arr = []
    for x, y, z in data:
        if x < 102 or x > 452:
            continue
        if not ((x, y, z) in law_dict.keys()):
            return False
        arr.append((x, y, z))

    arr = list(set(arr))
    arr.sort()

    for x in arr:
        if not (x in arr):
            return False
    return True


def analyze_crit(data, config):
    return accusation_dict[data[0]]


def analyze_law(data, config):
    return accusation_dict[data[0]]


def analyze_time(data, config):
    if data["sixing"]:
        return 0
    if data["wuqi"]:
        return 0
    v = 0
    if len(data["youqi"]) > 0:
        v = data["youqi"][-1]
    else:
        v = 0
    if v > 10 * 12:
        return 1
    if v > 7 * 12:
        return 2
    if v > 5 * 12:
        return 3
    if v > 3 * 12:
        return 4
    if v > 2 * 12:
        return 5
    if v > 1 * 12:
        return 6
    if v > 9:
        return 7
    if v > 6:
        return 8
    if v > 0:
        return 9
    return 10


word_dict = {}


def load(x, transformer):
    try:
        return transformer[x].astype(dtype=np.float32)
    except Exception as e:
        return transformer['UNK'].astype(dtype=np.float32)


def get_word_vec(x, config, transformer):
    vec = load(x, transformer)
    return vec


cnt1 = 0
cnt2 = 0


def format_sentence(data, config):
    result = data.split("。")
    for a in range(0, len(result)):
        temp = result[a].split("\t")
        result[a] = []
        for x in temp:
            if x != "":
                result[a].append(x)

    return result


def parse_sentence(data, config):
    global cnt1, cnt2
    # data = data.split("\t")
    result = data
    if result is None:
        return False

    result = format_sentence(data, config)

    if len(result) == 0:
        return False

    if len(result) > config.getint("data", "sentence_num"):
        cnt1 += 1
        # print("cnt1 %d" % cnt1)
        return False
    for a in range(0, len(result)):
        if len(result[a]) > config.getint("data", "sentence_len"):
            cnt2 += 1
            # print("cnt2 %d" % cnt2)
            return False

    return True


def generate_vector(data, config, transformer):
    vec = []
    len_vec = [0, 0]
    for x in data:
        temp_vec = []
        len_vec.append(len(x))
        len_vec[1] += 1
        for y in x:
            len_vec[0] += 1
            z = get_word_vec(y, config, transformer)
            temp_vec.append(z.to_list())
        while len(temp_vec) < config.getint("data", "sentence_len"):
            temp_vec.append(get_word_vec("BLANK", config, transformer).to_list())
        vec.append(temp_vec)

    temp_vec = []
    while len(temp_vec) < config.getint("data", "sentence_len"):
        temp_vec.append(get_word_vec("BLANK", config, transformer).to_list())

    while len(vec) < config.getint("data", "sentence_num"):
        vec.append(temp_vec)
        len_vec.append(1)
    if len_vec[1] > config.getint("data", "sentence_num"):
        gg
    for a in range(2, len(len_vec)):
        if len_vec[a] > config.getint("data", "sentence_len"):
            print(data)
            gg
    if len(len_vec) != config.getint("data", "sentence_num") + 2:
        gg

    return torch.stack(vec), torch.LongTensor(len_vec)

"""
def generate_vector(data, config, transformer):
    vec = []
    len_vec = [0, 0]
    for x in data:
        temp_vec = []
        len_vec.append(len(x))
        len_vec[1] += 1
        for y in x:
            len_vec[0] += 1
            z = get_word_vec(y, config, transformer)
            temp_vec.append(torch.from_numpy(z))
        while len(temp_vec) < config.getint("data", "sentence_len"):
            temp_vec.append(torch.from_numpy(get_word_vec("BLANK", config, transformer)))
        vec.append(torch.stack(temp_vec))

    temp_vec = []
    while len(temp_vec) < config.getint("data", "sentence_len"):
        temp_vec.append(torch.from_numpy(get_word_vec("BLANK", config, transformer)))

    while len(vec) < config.getint("data", "sentence_num"):
        vec.append(torch.stack(temp_vec))
        len_vec.append(1)
    if len_vec[1] > config.getint("data", "sentence_num"):
        gg
    for a in range(2, len(len_vec)):
        if len_vec[a] > config.getint("data", "sentence_len"):
            print(data)
            gg
    if len(len_vec) != config.getint("data", "sentence_num") + 2:
        gg

    return torch.stack(vec), torch.LongTensor(len_vec)
"""


def parse(data, config, transformer):
    label_list = config.get("data", "type_of_label").replace(" ", "").split(",")
    label = []
    for x in label_list:
        if x == "crit":
            label.append(analyze_crit(data["meta"]["crit"], config))
        if x == "law":
            label.append(analyze_law(data["meta"]["law"], config))
        if x == "time":
            label.append(analyze_time(data["meta"]["time"], config))
    vector, len_vec = generate_vector(data["content"], config, transformer)
    return vector, len_vec, torch.LongTensor(label)


def check(data, config):
    if len(data["meta"]["criminals"]) != 1:
        return False
    if len(data["meta"]["crit"]) == 0:
        return False
    if not (check_crit(data["meta"]["crit"])):
        return False
    if not (check_law(data["meta"]["law"])):
        return False

    return True
