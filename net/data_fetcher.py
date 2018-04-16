import os
import json

from net.data_formatter import parse, check
from net.utils import get_data_list


def create_dataset(file_list, config):
    dataset = []
    for file_name in file_list:
        file_path = os.path.join(config.get("data", "data_path"), str(file_name))
        if not (os.path.isfile(file_path)):
            continue
        print("Loading data from " + file_name + ".")
        cnt = 0
        f = open(file_path, "r")
        for line in f:
            data = json.loads(line)
            if check(data, config):
                if cnt % 10000 == 0:
                    print("Already load " + str(cnt) + " data...")
                dataset.append(parse(data, config))
                cnt += 1
        f.close()
        print("Loading " + str(cnt) + " data from " + file_name + " end.")

    return dataset  # DataLoader(dataset, batch_size=config.getint("data", "batch_size"),
    #         shuffle=config.getboolean("data", "shuffle"), drop_last=True, num_workers=4)


def init_train_dataset(config):
    return create_dataset(get_data_list(config.get("data", "train_data")), config)


def init_test_dataset(config):
    return create_dataset(get_data_list(config.get("data", "test_data")), config)


def init_dataset(config):
    train_dataset = init_train_dataset(config)
    test_dataset = init_test_dataset(config)

    return train_dataset, test_dataset
