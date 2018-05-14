import torch
import os
import json
import time
from torch.utils.data import DataLoader
import multiprocessing
import random
from torch.autograd import Variable

from net.data_formatter import check, parse, generate_vector, get_time_id, get_crit_id, get_law_id
from net.utils import get_data_list
from net.word2vec import word2vec
from net.utils import cut, print_info

# print("working...")
import h5py

transformer = None


def init_transformer(config):
    global transformer
    transformer = word2vec(os.path.join(config.get("data", "word2vec"), "word2id.pkl"),
                           os.path.join(config.get("data", "word2vec"), "vec_nor.npy"))
    transformer = {x: transformer.vec[y] for x, y in transformer.word2id.items()}

    print("Transformer init done")


"""duplicate_list = {
    "crit": {
        0: 5,
        1: 5,
        2: 10,
        3: 3,
        5: 15,
        7: 5,
        8: 7,
        9: 7,
        10: 40,
        11: 4,
        12: 4,
        14: 10,
        15: 5,
        16: 1,
        19: 3,
        20: 10,
        21: 3,
        22: 2,
        23: 2,
        24: 15,
        25: 7,
        27: 3,
        28: 2,
        29: 10,
        30: 5,
        31: 8,
        32: 20,
        33: 20,
        34: 7,
        35: 2,
        36: 4,
        37: 2,
        38: 2,
        39: 15
    },
    "law1": {
        0: 5,
        1: 5,
        2: 10,
        3: 3,
        6: 15,
        7: 5,
        8: 5,
        9: 5,
        10: 30,
        11: 4,
        12: 4,
        14: 10,
        15: 5,
        16: 1,
        19: 3,
        20: 10,
        21: 3,
        22: 2,
        23: 2,
        24: 20,
        25: 10,
        28: 5,
        29: 10,
        30: 10,
        31: 20,
        32: 20,
        33: 5,
        34: 2,
        35: 2,
        36: 2,
        37: 15
    },
    "time": {
        0: 20,
        1: 3,
        2: 3,
        3: 3
    }
}"""

duplicate_list = {"crit": {}, "law1": {}, "time": {}}


class reader():
    def __init__(self, file_list, config, num_process, train):
        self.file_list = file_list

        self.temp_file = None
        self.read_cnt = 0

        self.file_queue = multiprocessing.Queue()
        self.data_queue = multiprocessing.Queue()
        self.lock = multiprocessing.Lock()
        self.init_file_list(config)
        self.read_process = []
        self.num_process = num_process
        self.none_cnt = 0
        self.train = train

        for a in range(0, num_process):
            process = multiprocessing.Process(target=self.always_read_data,
                                              args=(config, self.data_queue, self.file_queue, a, transformer))
            self.read_process.append(process)
            self.read_process[-1].start()

    def init_file_list(self, config):
        if config.getboolean("data", "shuffle"):
            random.shuffle(self.file_list)
        for a in range(0, len(self.file_list)):
            self.file_queue.put(self.file_list[a])

    def always_read_data(self, config, data_queue, file_queue, idx, transformer):
        cnt = 40
        put_needed = False
        while True:
            if data_queue.qsize() < cnt:
                data = self.fetch_data_process(config, file_queue, transformer)
                if data is None:
                    if put_needed:
                        data_queue.put(data)
                        put_needed = False
                else:
                    data_queue.put(data)
                    # print(data_queue.qsize())
                    put_needed = True

    def gen_new_file(self, config, file_queue):
        if file_queue.qsize() == 0:
            self.temp_file = None
            return
        self.lock.acquire()
        try:
            p = file_queue.get(timeout=1)
            self.temp_file = open(os.path.join(config.get("data", "data_path"), config.get("data", "dataset"), p), "r")
            print_info("Loading file from " + str(p))
        except Exception as e:
            self.temp_file = None
        self.lock.release()

    def fetch_data_process(self, config, file_queue, transformer):
        batch_size = config.getint("data", "batch_size")

        data_list = []
        content_list = []

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
                    duplicate_time = 1
                    if self.train:
                        id1 = get_law_id(y["meta"]["law"], config)
                        id2 = get_crit_id(y["meta"]["crit"], config)
                        id3 = get_time_id(y["meta"]["time"], config)
                        if id1 in duplicate_list["law1"].keys():
                            duplicate_time += duplicate_list["law1"][id1]
                        if id2 in duplicate_list["crit"].keys():
                            duplicate_time += duplicate_list["crit"][id2]
                        if id3 in duplicate_list["time"].keys():
                            duplicate_time += duplicate_list["time"][id3]

                    while len(data_list) < batch_size and duplicate_time > 0:
                        duplicate_time -= 1
                        data_list.append(parse(y, config, transformer))
                        content_list.append(y["content"])
                        self.read_cnt += 1

            if len(data_list) < batch_size:
                return None

        dataloader = DataLoader(data_list[0:batch_size], batch_size=batch_size,
                                shuffle=config.getboolean("data", "shuffle"), drop_last=True)

        for idx, data in enumerate(dataloader):
            return data, content_list

        return None

    def fetch_data(self, config):
        # print("=================== %d ==================" % self.data_queue.qsize())
        while True:
            data = self.data_queue.get()
            if data is None:
                self.none_cnt += 1
                if self.none_cnt == self.num_process:
                    self.init_file_list(config)
                    self.none_cnt = 0
                    break
            else:
                break
        # print("done one")

        return data


def create_dataset(file_list, config, num_process, train):
    return reader(file_list, config, num_process, train)


def init_train_dataset(config):
    return create_dataset(get_data_list(config.get("data", "train_data")), config,
                          config.getint("train", "train_num_process"), True)


def init_test_dataset(config):
    return create_dataset(get_data_list(config.get("data", "test_data")), config,
                          config.getint("train", "test_num_process"), False)


def init_dataset(config):
    train_dataset = init_train_dataset(config)
    test_dataset = init_test_dataset(config)

    return train_dataset, test_dataset


def generate_article_list(config, usegpu):
    f = open("result/xf.txt", "r")
    xf_data = json.loads(f.readline())
    f = open("result/law_result1.txt", "r")
    law_list = []
    for line in f:
        arr = line.split(" ")
        tiao = int(arr[0])
        zhiyi = int(arr[1])
        # print(tiao,zhiyi)
        data = xf_data["[%d, %d]" % (tiao, zhiyi)]

        sentence = ""
        for x in data["tk"]:
            sentence += x["content"]
        sentence = sentence.replace(u"、", " ")

        sentence = sentence.split(u"。")
        sentence_temp = []
        for x in sentence:
            y = x.split(u"，")
            for z in y:
                sentence_temp.append(z)
        sentence = sentence_temp
        for a in range(0, len(sentence)):
            sentence[a] = cut(sentence[a])

        vec = generate_vector(sentence, config, transformer)
        if torch.cuda.is_available() and usegpu:
            vec = (Variable(vec[0].cuda()), Variable(vec[1].cuda()))
        else:
            vec = (Variable(vec[0]), Variable(vec[1]))

        # pdb.set_trace()

        law_list.append(vec)

    return law_list
