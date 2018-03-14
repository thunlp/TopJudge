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

num_file = 1
num_process = 1

word_case_list = [u"刑\s事", u"民\s事", u"行\s政", u"赔\s偿", u"执\s行"]
word_doc_list = [u"判决书", u"裁定书", u"调解书", u"决定书", u"通知书", u"批复", u"答复", u"函", u"令"]


def get_type_of_case(obj):
    if obj["content"] == "":
        return None

    for a in range(0, len(word_case_list)):
        match = re.search(word_case_list[a], obj["content"])
        if not (match is None):
            return a + 1

    return None


def get_type_of_doc(obj):
    if obj["Title"] == "":
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
        # try:
        data = json.loads(line)
        cnt += 1

        type1 = get_type_of_case(data["document"])
        type2 = get_type_of_doc(data["document"])
        if type1 is None or type2 is None:
            out_file = u"未知"
        else:
            out_file = word_case_list[type1].replace("\\s", "") + word_doc_list[type2]

        ouf_path = os.path.join(out_path, out_file)
        os.makedirs(out_path)
        ouf_path = os.path.join(out_path, str(file_num))
        ouf = open(ouf_path, "a")
        print(json.dumps(data, ensure_ascii=False), file=ouf)

        if cnt % 50 == 0:
            break
            print(in_path, cnt)
            # break

            # except Exception as e:
            #    print(e)
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
