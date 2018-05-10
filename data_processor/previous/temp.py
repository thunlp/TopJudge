# coding: UTF-8

import os
import json
import re


in_path = "/disk/mysql/law_data/final_data/"
out_path = "/disk/mysql/law_data/temp_data/"
mid_text = u"\t"
num_process = 4
num_file = 20

accusation_file = "/home/zhx/law_pre/data_processor/accusation_list2.txt"
f = open(accusation_file, "r")
accusation_list = json.loads(f.readline())
for a in range(0, len(accusation_list)):
    accusation_list[a] = accusation_list[a].replace('[', '').replace(']', '')
f.close()


able_list = [248,247,201]

def draw_out(in_path, out_path,cntx):
    print(in_path)
    inf = open(in_path, "r")

    cnt = 0
    cx = 0
    for line in inf:
        try:
            data = json.loads(line)
            for idx in able_list:
                if idx in data["meta"]["crit"]:
                    outf_path = os.path.join(out_path,accusation_list[idx],str(cntx))
                    ouf = open(outf_path, "a")
                    print(json.dumps(data),file=ouf)


            cnt += 1
            if cnt % 50000 == 0:
                print(in_path, cnt, cx)
                # break

        except Exception as e:
            pass  # print(e)
            gg


def work(from_id, to_id):
    for a in range(int(from_id), int(to_id)):
        print(str(a) + " begin to work")
        draw_out(os.path.join(in_path, str(a)),out_path,a)
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
