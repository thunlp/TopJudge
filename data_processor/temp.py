# coding: UTF-8

import os
import json
import re


in_path = "/disk/mysql/law_data/final_data/"
out_path = "/disk/mysql/law_data/temp_data/"
mid_text = u"\t"
num_process = 1
num_files = 20

accusation_file = "/home/zhx/law_pre/data_processor/accusation_list2.txt"
f = open(accusation_file, "r")
accusation_list = json.loads(f.readline())
for a in range(0, len(accusation_list)):
    accusation_list[a] = accusation_list[a].replace('[', '').replace(']', '')
f.close()

for a in range(0,len(accusation_list)):
    print(a,accusation_list[a])


def draw_out(in_path, out_path):
    print(in_path)
    inf = open(in_path, "r")
    ouf = open(out_path, "w")

    cnt = 0
    cx = 0
    for line in inf:
        #try:
            data = json.loads(line)

            cnt += 1
            if cnt % 50 == 0:
                gg
                print(in_path, cnt, cx)
                # break

        #except Exception as e:
        #    pass  # print(e)
            # gg


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
