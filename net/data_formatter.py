import os
import json
import torch
import random
import numpy as np

from net.loader import accusation_dict, accusation_list, law_dict, law_list
from net.loader import get_num_classes


def check_crit(data):
    cnt = 0
    for x in data:
        if x in accusation_dict.keys():
            cnt += 1
        else:
            return False
    return cnt == 1


def check_law(data):
    arr = []
    for x, y, z in data:
        if x < 102 or x > 452:
            continue
        if not ((x, y) in law_dict.keys()):
            return False
        arr.append((x, y))

    arr = list(set(arr))
    arr.sort()

    cnt = 0
    for x in arr:
        if x in arr:
            cnt += 1  # return False
    return cnt == 1


def get_crit_id(data, config):
    for x in data:
        if x in accusation_dict.keys():
            return accusation_dict[x]


def get_law_id(data, config):
    for x in data:
        y = (x[0], x[1])
        if y in law_dict.keys():
            return law_dict[y]


def get_time_id(data, config):
    v = 0
    if len(data["youqi"]) > 0:
        v1 = data["youqi"][-1]
    else:
        v1 = 0
    if len(data["guanzhi"]) > 0:
        v2 = data["guanzhi"][-1]
    else:
        v2 = 0
    if len(data["juyi"]) > 0:
        v3 = data["juyi"][-1]
    else:
        v3 = 0
    v = max(v1, v2, v3)

    if data["sixing"]:
        opt = 0
    elif data["wuqi"]:
        opt = 0
    elif v > 10 * 12:
        opt = 1
    elif v > 7 * 12:
        opt = 2
    elif v > 5 * 12:
        opt = 3
    elif v > 3 * 12:
        opt = 4
    elif v > 2 * 12:
        opt = 5
    elif v > 1 * 12:
        opt = 6
    elif v > 9:
        opt = 7
    elif v > 6:
        opt = 8
    elif v > 0:
        opt = 9
    else:
        opt = 10

    return opt


def analyze_crit(data, config):
    res = torch.from_numpy(np.zeros(get_num_classes("crit")))
    for x in data:
        if x in accusation_dict.keys():
            res[accusation_dict[x]] = 1
    return res


def analyze_law(data, config):
    res = torch.from_numpy(np.zeros(get_num_classes("law")))
    for x in data:
        y = (x[0], x[1])
        if y in law_dict.keys():
            res[law_dict[y]] = 1
    return res


def analyze_time(data, config):
    res = torch.from_numpy(np.zeros(get_num_classes("time")))

    opt = get_time_id(data, config)

    res[opt] = 1
    return res


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


def check_sentence(data, config):
    if len(data) > config.getint("data", "sentence_num"):
        return False
    for x in data:
        if len(x) > config.getint("data", "sentence_len"):
            return False
    return True


def generate_vector(data, config, transformer):
    vec = []
    len_vec = [0, 0]
    blank = torch.from_numpy(get_word_vec("BLANK", config, transformer))
    for x in data:
        temp_vec = []
        len_vec.append(len(x))
        len_vec[1] += 1
        for y in x:
            len_vec[0] += 1
            z = get_word_vec(y, config, transformer)
            temp_vec.append(torch.from_numpy(z))
        while len(temp_vec) < config.getint("data", "sentence_len"):
            temp_vec.append(blank)
        vec.append(torch.stack(temp_vec))

    temp_vec = []
    while len(temp_vec) < config.getint("data", "sentence_len"):
        temp_vec.append(blank)

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
    return vector, len_vec, torch.cat(label)


def check(data, config):
    if not (check_sentence(data["content"], config)):
        return False
    if len(data["meta"]["criminals"]) != 1:
        return False
    if len(data["meta"]["crit"]) == 0 or len(data["meta"]["law"]) == 0:
        return False
    if not (check_crit(data["meta"]["crit"])):
        return False
    if not (check_law(data["meta"]["law"])):
        return False

    return True
