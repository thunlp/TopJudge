# coding: UTF-8

import os
import json

in_path = "/disk/mysql/law_data/critical_data/"
out_path = "/disk/mysql/law_data/final_data/"
mid_text = u"  _(:з」∠)_  "
title_list = ["docId", "caseNumber", "caseName", "spcx", "court", "time", "caseType", "bgkly", "yuanwen", "document",
              "cause", "docType", "keyword", "lawyer", "punishment", "result", "judge"]

num_process = 1
num_file = 1


def draw_out(in_path, out_path):
    print(in_path)
    inf = open(in_path, "r")
    ouf = open(out_path, "w")

    cnt = 0
    cx = 0
    for line in inf:
        try:
            data = json.loads(line)
            if "AJJBQK" in data["document"] or "SSJL" in data["document"]:
                if "AJJBQK" in data["document"]:
                    print(len(data["document"]["AJJBQK"]),cx)
                    if cx == 1:
                        print(data["document"]["AJJBQK"])
                if "SSJL" in data["document"]:
                    print(len(data["document"]["SSJL"]),cx)
                    if cx == 11:
                        print(data["document"]["SSJL"])
                cx += 1

            cnt += 1
            if cnt == 50:
                print(in_path, cnt, cx)
                break

        except Exception as e:
            print(e)
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
