# coding: UTF-8

import os
import json

in_path = "/disk/mysql/law_data/origin_split/"
out_path = "/disk_mysql/law_data/formed_data"
mid_text = u"  _(:з」∠)_  "
title_list = ["docId", "caseNumber", "caseName", "spcx", "court", "time", "caseType", "bgkly", "yuanwen", "document",
              "cause", "docType", "keyword", "lawyer", "punishment", "result", "judge"]

print(len(title_list))

gg


def draw_out(in_path, out_path):
    inf = open(in_path, "r")
    ouf = open(out_path, "w")

    temped = None
    data_str = None
    for line in inf:
        try:
            cnt = line.count(mid_text)
            if cnt == 16:
                data_str = line[:-1]
            elif cnt == 9:
                temped = line[:-1]
                continue
            elif cnt == 7:
                if not (temped is None):
                    data_str = temped + line[:-1]
                else:
                    continue
            else:
                temped = None
                continue

            data_str = data_str.split(mid_text)

            data = {}

            if len(data_str) == len(title_list):
                for a in range(0, len(data_str)):
                    data[title_list[a]] = data_str[a]
                for a in data:
                    if data[a] == u"\\N":
                        data[a] = ""
                if data["document"] == "":
                    data["docuemnt"] = "{\"content\":\"\"}"
                data["documnet"] = json.loads(data["document"])

                print(json.dumps(data),file=ouf)

                break
            else:
                gg

        except Exception as e:
            print(e)
            gg


def work(from_id, to_id):
    for a in range(from_id, to_id):
        print(str(a) + " begin to work")
        draw_out(os.path.join(in_path, str(a)), os.path.join(out_path, str(a)))
        print(str(a) + " work done")


num_file = 1
num_process = 1

if __name__ == "__main__":
    gg

    import multiprocessing

    process_pool = []

    for a in range(0, num_process):
        process_pool.append(
            multiprocessing.Process(target=work, args=(a * num_file / num_process, (a + 1) * num_file / num_process)))

    for a in process_pool:
        a.start()

    for a in process_pool:
        a.join()
