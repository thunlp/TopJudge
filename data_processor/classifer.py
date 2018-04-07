# coding: UTF-8

import os
import json
import re

# in_path = r"D:\work\law_pre\test\in"
# out_path = r"D:\work\law_pre\test\out"
in_path = r"/disk/mysql/law_data/formed_data"
out_path = r"/disk/mysql/law_data/classified_data"
mid_text = u"  _(:з」∠)_  "
title_list = ["docId", "caseNumber", "caseName", "spcx", "court", "time", "caseType", "bgkly", "yuanwen", "document",
              "cause", "docType", "keyword", "lawyer", "punishment", "result", "judge"]

num_file = 20
num_process = 4

word_case_list = [r"刑(\s?)事", r"民(\s?)事", r"行(\s?)政", r"赔(\s?)偿", r"执(\s?)行"]
word_doc_list = [r"判(\s?)决(\s?)书", r"裁(\s?)定(\s?)书", r"调(\s?)解(\s?)书", r"决(\s?)定(\s?)书", r"通(\s?)知(\s?)书", r"批复", r"答复",
                 r"函", r"令"]


def get_type_of_case(obj):
    if not ("Title" in obj) or obj["Title"] == "":
        return None

    for a in range(0, len(word_case_list)):
        match = re.search(word_case_list[a], obj["Title"])
        if not (match is None):
            return a

    return None


def get_type_of_doc(obj):
    if not ("Title" in obj) or obj["Title"] == "":
        return None

    for a in range(0, len(word_doc_list)):
        match = re.search(word_doc_list[a], obj["Title"])
        if not (match is None):
            return a

    # print obj["Title"]

    return None


def draw_out(in_path, file_num):
    inf = open(in_path, "r")
    print(in_path)

    cnt = 0
    for line in inf:
        try:
            data = json.loads(line)
            cnt += 1

            type1 = get_type_of_case(data["document"])
            type2 = get_type_of_doc(data["document"])
            if type1 is None or type2 is None:
                out_file = u"未知"
            else:
                out_file = word_case_list[type1].replace("\\s", "").replace("(","").replace(")","").replace("?","") + word_doc_list[type2].replace("\\s","").replace("(","").replace(")","").replace("?","")

            ouf_path = os.path.join(out_path, out_file)
            if not (os.path.exists(ouf_path)):
                try:
                    os.makedirs(ouf_path)
                except Exception as e:
                    pass
            ouf_path = os.path.join(ouf_path, str(file_num))
            ouf = open(ouf_path, "a")
            print(json.dumps(data, ensure_ascii=False), file=ouf)

            if cnt % 50000 == 0:
                print(in_path, cnt)

                # break

        except Exception as e:
            print(e)
            # gg


def work(from_id, to_id):
    for a in range(int(from_id), int(to_id)):
        print(str(a) + " begin to work")
        draw_out(os.path.join(in_path, str(a)), a)
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
