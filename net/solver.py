# coding: UTF-8

import os
import json
import configparser
import torch
import numpy as np

configFilePath = "../config/multi_lstm_crit_baseline_small.config"
config = configparser.RawConfigParser()
config.read(configFilePath)

in_path = "/data/disk1/private/zhonghaoxi/law/data"
out_path = "/data/disk1/private/zhonghaoxi/law/format_data"

num_process = 1
num_file = 20

from data_formatter import parse, check


def draw_out(in_path, out_path):
    print(in_path)
    inf = open(in_path, "r")

    cnt = 0
    cx = 0
    res = []
    for line in inf:
        # try:
        data = json.loads(line)
        if check(data, config):
            a, b, c = parse(data, config)
            # res[0].append(a)
            # res[1].append(b)
            # res[2].append(c)
            res.append((a.numpy(), b.numpy(), c.numpy()))
            cnt += 1
            if cnt % 5000 == 0:
                print(in_path, cnt, cx)
                # break

                # except Exception as e:
                # pass  # print(e)
                # gg

    np.save(out_path, np.array(res))
    # np.save(out_path + "-inputs", torch.stack(res[0]).numpy())
    # np.save(out_path + "-doc_len", torch.stack(res[1]).numpy())
    # np.save(out_path + "-labels", torch.stack(res[2]).numpy())


def work(from_id, to_id):
    for a in range(int(from_id), int(to_id)):
        print(str(a) + " begin to work")
        draw_out(os.path.join(in_path, str(a)), os.path.join(out_path, str(a)))
        print(str(a) + " work done")


if __name__ == "__main__":
    import multiprocessing

    process_pool = []

    for a in range(0, num_process):
        process_pool.append(
            multiprocessing.Process(target=work, args=(a * num_file / num_process, (a + 1) * num_file / num_process)))

    for a in process_pool:
        a.start()

    for a in process_pool:
        a.join()
