import os
import json
import torch
from word2vec import word2vec
import random

transformer = word2vec()
accusation_list = []
accusation_dict = {}
f = open("result/result_bac.txt", "r")
for line in f:
    data = int(line[:-1].replace("\n", "").split(" ")[1])
    accusation_list.append(data)
    accusation_dict[data] = len(accusation_list) - 1
print(accusation_list)
print(accusation_dict)


def get_num_classes(s):
    if s == "crit":
        return 50
    if s == "law":
        return 4
    if s == "time":
        return gg
    gg


def get_data_list(d):
    return d.replace(" ", "").split(",")


def analyze_crit(data, config):
    return accusation_dict[data[0]]


def analyze_law(data, config):
    pass


def analyze_time(data, config):
    # print(data)
    # gg
    if data["sixing"]:
        return 0
    if data["wuqi"]:
        return 1
    v = 0
    for x in data["youqi"]:
        v = max(x, v)
    for x in data["juyi"]:
        v = max(x, v)
    for x in data["guanzhi"]:
        v = max(x, v)
    if v > 25 * 12:
        return 2
    if v > 15 * 12:
        return 3
    if v > 10 * 12:
        return 4
    if v > 7 * 12:
        return 5
    if v > 5 * 12:
        return 6
    if v > 3 * 12:
        return 7
    if v > 2 * 12:
        return 8
    if v > 1 * 12:
        return 9
    if v > 9:
        return 10
    if v > 6:
        return 11
    if v > 0:
        return 12
    return 13


word_dict = {}


def get_word_vec(x, config):
    # if not (x in word_dict):
    #    word_dict[x] = torch.rand(config.getint("data", "vec_size"))
    vec = transformer.load(x)
    # print(type(vec))
    return vec

    # return word_dict[x], True


def generate_vector(data, config):
    data = data.split("\t")
    vec = []
    for x in data:
        y = get_word_vec(x, config)
        vec.append(torch.from_numpy(y))
        if len(vec) == config.getint("data", "pad_length"):
            break
    len_vec = len(vec)
    while len(vec) < config.getint("data", "pad_length"):
        vec.append(torch.from_numpy(transformer.load("BLANK")))

    # print(torch.stack(vec))
    return torch.stack([torch.stack(vec)]), len_vec


def parse(data, config):
    label_list = config.get("data", "type_of_label").replace(" ", "").split(",")
    label = []
    for x in label_list:
        if x == "crit":
            label.append(analyze_crit(data["meta"]["crit"], config))
        if x == "law":
            label.append(analyze_law(data["meta"]["law"], config))
        if x == "time":
            label.append(analyze_time(data["meta"]["time"], config))
    vector, len_vec = generate_vector(data["content"], config)
    return vector, len_vec, torch.LongTensor(label)


def check(data, config):
    if len(data["meta"]["crit"]) > 1 or len(data["meta"]["crit"]) == 0:
        return False
    if not (int(data["meta"]["crit"][0]) in accusation_dict):
        return False
    if len(data["content"].split("\t")) > config.getint("data", "pad_length"):
        return False

    return True
