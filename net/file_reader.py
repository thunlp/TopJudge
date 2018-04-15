import random
import os
import json
import time
from utils import get_data_list
from torch.utils.data import DataLoader
import multiprocessing
from word2vec import word2vec

from data_formatter import check, parse

# print("working...")
import h5py

transformer = word2vec()
# manager = multiprocessing.Manager()
transformer = {x: transformer.vec[y] for x, y in transformer.word2id.items()}
# transformer = None
# print(len(transformer))
# transformer = manager.list([transformer])

print("working done")

train_num_process = 5
test_num_process = 1


class reader():
    def __init__(self, file_list, config, num_process):
        self.file_list = file_list

        self.temp_file = None
        self.read_cnt = 0

        self.file_queue = multiprocessing.Queue()
        self.data_queue = multiprocessing.Queue()
        self.init_file_list()
        self.read_process = []
        for a in range(0, num_process):
            process = multiprocessing.Process(target=self.always_read_data,
                                              args=(config, self.data_queue, self.file_queue, a, transformer))
            self.read_process.append(process)
            self.read_process[-1].start()

    def init_file_list(self):
        for a in range(0, len(self.file_list)):
            self.file_queue.put(self.file_list[a])

    def always_read_data(self, config, data_queue, file_queue, idx, transformer):
        # transformer = h5py.File('/data/disk1/private/zhonghaoxi/law/word2vec/data.h5','r')
        cnt = 20
        put_needed = False
        while True:
            if data_queue.qsize() < cnt:
                data = self.fetch_data_process(config, file_queue, transformer)
                if data is None:
                    if put_needed and idx == 0:
                        data_queue.put(data)
                    put_needed = False
                else:
                    data_queue.put(data)
                    put_needed = True

    def gen_new_file(self, config, file_queue):
        if file_queue.qsize() == 0:
            self.temp_file = None
            return
        try:
            p = file_queue.get(timeout=1)
            self.temp_file = open(os.path.join(config.get("data", "data_path"), p), "r")
            print("Loading file from " + str(p))
        except Exception as e:
            self.temp_file = None

    def fetch_data_process(self, config, file_queue, transformer):
        batch_size = config.getint("data", "batch_size")

        data_list = []

        if batch_size > len(data_list):
            if self.temp_file is None:
                self.gen_new_file(config, file_queue)
                if self.temp_file is None:
                    return None

            while len(data_list) < batch_size:
                x = self.temp_file.readline()
                if x == "" or x is None:
                    self.gen_new_file(config, file_queue)
                    if self.temp_file is None:
                        return None
                    continue

                try:
                    y = json.loads(x)
                except Exception as e:
                    print("==========================")
                    print(x)
                    print("==========================")
                    gg
                if check(y, config):
                    data_list.append(parse(y, config, transformer))
                    self.read_cnt += 1

            if len(data_list) < batch_size:
                return None

        dataloader = DataLoader(data_list[0:batch_size], batch_size=batch_size,
                                shuffle=config.getboolean("data", "shuffle"), drop_last=True)

        for idx, data in enumerate(dataloader):
            return data

        return None

    def fetch_data(self, config):
        # print("=================== %d ==================" % self.data_queue.qsize())
        data = self.data_queue.get()
        if data is None:
            self.init_file_list()
        # print("done one")

        return data


def create_dataset(file_list, config, num_process):
    return reader(file_list, config, num_process)


def init_train_dataset(config):
    return create_dataset(get_data_list(config.get("data", "train_data")), config, train_num_process)


def init_test_dataset(config):
    return create_dataset(get_data_list(config.get("data", "test_data")), config, test_num_process)


def init_dataset(config):
    train_dataset = init_train_dataset(config)
    test_dataset = init_test_dataset(config)

    return train_dataset, test_dataset
