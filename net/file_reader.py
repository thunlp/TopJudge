import random
import os
import json
from data_formatter import parse, check, get_data_list, get_num_classes


class reader():
    def __init__(self, file_list):
        self.file_list = file_list
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
        p = random.randint(0, len(self.file_list) - 1)
        while self.use_list[p]:
            p = random.randint(0, len(self.file_list) - 1)

        self.use_list[p] = True

        self.temp_file = open(os.path.join(config.get("data", "data_path"), str(self.file_list[p])), "r")

    def fetch_data(self, config):
        batch_size = config.getint("data", "batch_size")

        if batch_size > len(self.data_list):
            if self.temp_file is None:
                self.gen_new_file(config)

            while len(self.data_list) < 4 * batch_size:
                now_line = self.temp_file.readline()
                if now_line == '':
                    if self.rest == 0:
                        break
                    self.gen_new_file(config)
                    continue
                data = json.loads(now_line)
                if check(data, config):
                    self.data_list.append(parse(data, config))

            if len(self.data_list) < batch_size:
                for a in range(0, len(self.file_list)):
                    self.use_list[a] = False
                self.data_list = []
                self.rest = len(self.file_list)
                self.temp_file = None
                return None

        dataloader = DataLoader(self.data_list[0:batch_size], batch_size=batch_size,
                                shuffle=config.getboolean("data", "shuffle"))
        self.data_list = self.data_list[batch_size:len(self.data_list) - 1]
        for idx, data in enumerate(dataloader):
            return data

        return None


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
