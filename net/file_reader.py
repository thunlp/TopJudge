import os
import json
import torch
from word2vec import word2vec
import random

transformer = word2vec()


def get_num_classes(s):
    if s == "crit":
        return 3
    if s == "law":
        return 4
    if s == "time":
        return 14
    gg


def get_data_list(d):
    return d.replace(" ", "").split(",")


def analyze_crit(data, config):
    return data[0]


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
    if len(data["meta"]["crit"]) > 1:
        return False

    return True


class reader():
    def __init__(self, file_list):
        self.file_list = []
        self.use_list = []
        for a in range(0, len(file_list)):
            self.use_list.append(False)
        self.data_list = []
        self.temp_file = None
        self.rest = len(self.file_list)

    def gen_new_file(self, config):
        if self.rest == 0:
            return
        self.rest -= 1
        p = random.randint(0, len(self.file_list))
        while self.use_list[p]:
            p = random.randint(0, len(self.file_list))

        self.use_list[p] = True

        self.temp_file = open(os.path.join(config.get("data", "data_path"), str(self.file_list[p])), "r")

    def fetch_data(self, config):
        batch_size = config.getint("data", "batch_size")

        if batch_size > len(self.data_list):
            if self.temp_file is None:
                self.gen_new_file()

            while len(self.data_list) < 4 * batch_size:
                now_line = self.temp_file.readline()
                if now_line == '':
                    break
                data = json.loads(now_line)
                if check(data, config):
                    self.data_list.append(parse(data, config))

            if len(self.data_list) < batch_size:
                return None

        data = torch.stack(self.data[0:batch_size])
        self.data_list = self.data_list[batch_size:-1]

        return data


def create_dataset(file_list, config):
    return reader(file_list)


def init_train_dataset(config):
    return create_dataset(get_data_list(config.get("data", "train_data")), config)


def init_test_dataset(config):
    return create_dataset(get_data_list(config.get("data", "test_data")), config)


def init_dataset(config):
    train_dataset = init_train_dataset(config)
    test_dataset = init_test_dataset(config)

    return train_dataset, test_dataset
