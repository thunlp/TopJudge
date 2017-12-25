import os
import json
from torch.utils.data import DataLoader
import torch


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
    #print(data)
    #gg
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
    if not (x in word_dict):
        word_dict[x] = torch.rand(config.getint("data", "vec_size"))

    return word_dict[x], True


def generate_vector(data, config):
    data = data.split("\t")
    vec = []
    for x in data:
        y, z = get_word_vec(x, config)
        if z:
            vec.append(y)

    while len(vec) < config.getint("data", "pad_length"):
        vec.append(torch.FloatTensor(config.getint("data","vec_size")))

    return torch.stack([torch.stack(vec)])


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
    vector = generate_vector(data["content"], config)
    return vector, torch.LongTensor(label)


def check(data, config):
    if len(data["meta"]["crit"]) > 1:
        return False
    if len(data["meta"]["law"]) > 1:
        return False

    return True


def create_loader(file_list, config):
    dataset = []
    for file_name in file_list:
        print("Loading data from "+file_name+".")
        file_path = os.path.join(config.get("data", "data_path"), str(file_name))
        cnt = 0
        f = open(file_path, "r")
        for line in f:
            data = json.loads(line)
            if check(data, config):
                if cnt % 10000 == 0:
                    print("Already load "+str(cnt)+" data...")
                dataset.append(parse(data, config))
                cnt += 1
        f.close()
        print("Loading " + str(cnt)+ " data from "+file_name + " end.")

    return DataLoader(dataset, batch_size=config.getint("data", "batch_size"),
                      shuffle=config.getboolean("data", "shuffle"))


def init_train_loader(config):
    return create_loader(get_data_list(config.get("data", "train_data")), config)


def init_test_loader(config):
    return create_loader(get_data_list(config.get("data", "test_data")), config)


def init_loader(config):
    train_loader = init_train_loader(config)
    test_loader = init_test_loader(config)

    return train_loader, test_loader
