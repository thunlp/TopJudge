import random
import os
import json
import time
from utils import get_data_list
from torch.utils.data import DataLoader
import multiprocessing
from word2vec import word2vec

from data_formatter import check, parse

transformer = word2vec()

num_process = 4


class reader():
    def __init__(self, file_list, config):
        self.file_list = file_list

        self.temp_file = None
        self.read_cnt = 0

        self.file_queue = multiprocessing.Queue()
        self.data_queue = multiprocessing.Queue()
        for a in range(0, num_process):
            self.read_process = multiprocessing.Process(target=self.always_read_data,
                                                        args=(config, self.data_queue, self.file_queue, transformer, a))
            self.read_process.start()

    def init_file_list(self):
        for a in range(0, len(self.file_list)):
            self.file_queue.put(self.file_list[a])

    def always_read_data(self, config, data_queue, file_queue, transformer, idx):
        cnt = 10
        put_needed = False
        while True:
            if data_queue.qsize() < cnt:
                data = self.fetch_data_process(self, config, file_queue, transformer)
                if data is None:
                    if put_needed:
                        data_queue.put(data)
                    put_needed = False
                else:
                    data_queue.put(data)
                    put_needed = True

    def gen_new_file(self, config, file_queue):
        if self.rest == 0:
            return
        try:
            p = file_queue.get(timeout=1)
            self.temp_file = open(os.path.join(config.get("data", "data_path"), str(self.file_list[p])), "r")
            print("Loading file from " + str(self.file_list[p]))
        except Exception as e:
            self.temp_file = None

    def fetch_data_process(self, config, file_queue, transformer):
        batch_size = config.getint("data", "batch_size")

        data_list = []

        if batch_size > len(data_list):
            if self.temp_file is None:
                self.gen_new_file(config)
                if self.temp_file is None:
                    return None

            while len(data_list) < batch_size:
                x = self.temp_file.readline()
                if x == "":
                    self.gen_new_file(config, file_queue)
                    if self.temp_file is None:
                        return None

                y = json.loads(x)
                if check(y, config):
                    data_list.append(parse(y, config, transformer))
                    self.read_cnt += 1

            if len(data_list) < batch_size:
                return None

        dataloader = DataLoader(self.data_list[0:batch_size], batch_size=batch_size,
                                shuffle=config.getboolean("data", "shuffle"), drop_last=True)

        for idx, data in enumerate(dataloader):
            return data

        return None

    def fetch_data(self, config):
        print("=================== %d ==================" % self.queue.qsize())
        data = self.queue.get()
        if data is None:
            self.init_file_list()

        return data


def create_dataset(file_list, config):
    return reader(file_list, config)


def init_train_dataset(config):
    return create_dataset(get_data_list(config.get("data", "train_data")), config)


def init_test_dataset(config):
    return create_dataset(get_data_list(config.get("data", "test_data")), config)


def init_dataset(config):
    train_dataset = init_train_dataset(config)
    test_dataset = init_test_dataset(config)

    return train_dataset, test_dataset
