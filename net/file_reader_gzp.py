import random
import os
import json
from data_formatter_gzp import parse, check, get_data_list
from torch.utils.data import DataLoader


class reader():
    def __init__(self, file_list):
        self.file_list = file_list
        self.use_list = []
        for a in range(0, len(file_list)):
            self.use_list.append(False)
        self.data_list = []
        self.temp_file = None
        self.rest = len(self.file_list)
        self.read_cnt = 0

    def gen_new_file(self, config):
        if self.rest == 0:
            return
        print("Already loaded %d data" % self.read_cnt)
        self.read_cnt = 0
        self.rest -= 1
        p = random.randint(0, len(self.file_list) - 1)
        while self.use_list[p]:
            p = random.randint(0, len(self.file_list) - 1)

        self.use_list[p] = True
        print("Loading file from " + str(self.file_list[p]))

        self.temp_file = open(os.path.join(config.get("data", "data_path"), str(self.file_list[p])), "r")
        cnt = 0
        #while cnt < 8192 + 1152 + 62:
        #    x = self.temp_file.readline()
        #    if check(json.loads(x),config):
        #        cnt += 1

    def fetch_data(self, config):
        batch_size = config.getint("data", "batch_size")

        if batch_size > len(self.data_list):
            if self.temp_file is None:
                self.gen_new_file(config)

            while len(self.data_list) < batch_size:
                x = self.temp_file.readline()
                if x == "":
                    if self.rest == 0:
                        break
                    self.gen_new_file(config)
                    continue
                y = json.loads(x)
                if check(y, config):
                    self.data_list.append(parse(y, config))
                    self.read_cnt += 1

                # gg

            if len(self.data_list) < batch_size:
                for a in range(0, len(self.file_list)):
                    self.use_list[a] = False
                self.data_list = []
                self.rest = len(self.file_list)
                self.temp_file = None
                return None

        dataloader = DataLoader(self.data_list[0:batch_size], batch_size=batch_size,
                                shuffle=config.getboolean("data", "shuffle"), drop_last=True)
        self.data_list = []
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
