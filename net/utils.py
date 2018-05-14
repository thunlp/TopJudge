import torch
from torch.autograd import Variable
import json
import thulac
import pdb
import time

from net.loader import get_name

cutter = None


def init_thulac(config):
    global cutter
    cutter = thulac.thulac(model_path=config.get("data","thulac"), seg_only=True, filt=False)


def print_time():
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


def print_info(s):
    times = str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print("[%s] %s" % (times, s))


def get_data_list(d):
    return d.replace(" ", "").split(",")


def calc_accuracy(outputs, labels, loss_type, res):
    if loss_type == "multi_classification":
        if len(labels[0]) != len(outputs[0]):
            raise ValueError('Input dimensions of labels and outputs must match.')

        outputs = outputs.data
        labels = labels.data

        nr_classes = outputs.size(1)
        for i in range(nr_classes):
            outputs1 = (outputs[:, i] >= 0.5).long()
            labels1 = (labels[:, i] >= 0.5).long()
            res[i]["TP"] += int((labels1 * outputs1).sum())
            res[i]["FN"] += int((labels1 * (1 - outputs1)).sum())
            res[i]["FP"] += int(((1 - labels1) * outputs1).sum())
            res[i]["TN"] += int(((1 - labels1) * (1 - outputs1)).sum())

        return res

    elif loss_type == "single_classification":
        id1 = torch.max(outputs, dim=1)[1]
        id2 = torch.max(labels, dim=1)[1]
        for a in range(0, len(id1)):
            it_is = int(id1[a])
            should_be = int(id2[a])
            if it_is == should_be:
                res[it_is]["TP"] += 1
            else:
                res[it_is]["FP"] += 1
                res[should_be]["FN"] += 1

        return res


def get_value(res):
    # According to https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
    if res["TP"] == 0:
        if res["FP"] == 0 and res["FN"] == 0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
    else:
        precision = 1.0 * res["TP"] / (res["TP"] + res["FP"])
        recall = 1.0 * res["TP"] / (res["TP"] + res["FN"])
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def gen_result(res, test=False, file_path=None, class_name=None):
    precision = []
    recall = []
    f1 = []
    total = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for a in range(0, len(res)):
        total["TP"] += res[a]["TP"]
        total["FP"] += res[a]["FP"]
        total["FN"] += res[a]["FN"]
        total["TN"] += res[a]["TN"]

        p, r, f = get_value(res[a])
        precision.append(p)
        recall.append(r)
        f1.append(f)

    micro_precision, micro_recall, micro_f1 = get_value(total)

    macro_precision = 0
    macro_recall = 0
    macro_f1 = 0
    for a in range(0, len(f1)):
        macro_precision += precision[a]
        macro_recall += recall[a]
        macro_f1 += f1[a]

    macro_precision /= len(f1)
    macro_recall /= len(f1)
    macro_f1 /= len(f1)

    print("Micro precision\t%.3f" % micro_precision)
    print("Macro precision\t%.3f" % macro_precision)
    print("Micro recall\t%.3f" % micro_recall)
    print("Macro recall\t%.3f" % macro_recall)
    print("Micro f1\t%.3f" % micro_f1)
    print("Macro f1\t%.3f" % macro_f1)

    if test and not (file_path is None):
        f = open(file_path, "w")
        print("Micro precision\t%.3f" % micro_precision, file=f)
        print("Macro precision\t%.3f" % macro_precision, file=f)
        print("Micro recall\t%.3f" % micro_recall, file=f)
        print("Macro recall\t%.3f" % macro_recall, file=f)
        print("Micro f1\t%.3f" % micro_f1, file=f)
        print("Macro f1\t%.3f" % macro_f1, file=f)
        print("", file=f)
        total_cnt = 0
        for a in range(0, len(res)):
            total_cnt += res[a]["TP"] + res[a]["FN"]
        print(total_cnt, file=f)
        for a in range(0, len(res)):
            temp = res[a]
            temp["total"] = temp["TP"] + temp["FN"]
            temp["precision"], temp["recall"], temp["f1"] = get_value(temp)
            if not (class_name is None):
                print("%d %s " % (a, get_name(class_name, a)), temp, file=f)
            else:
                print("%d " % a, res[a], file=f)
        f.close()

    print("")


def generate_graph(config):
    s = config.get("data", "graph")
    arr = s.replace("[", "").replace("]", "").split(",")
    graph = []
    n = 0
    if (s == "[]"):
        arr = []
        n = 3
    for a in range(0, len(arr)):
        arr[a] = arr[a].replace("(", "").replace(")", "").split(" ")
        arr[a][0] = int(arr[a][0])
        arr[a][1] = int(arr[a][1])
        n = max(n, max(arr[a][0], arr[a][1]))

    n += 1
    for a in range(0, n):
        graph.append([])
        for b in range(0, n):
            graph[a].append(False)

    for a in range(0, len(arr)):
        graph[arr[a][0]][arr[a][1]] = True

    return graph


def cut(s):
    data = cutter.cut(s)
    result = []
    first = True
    for x, y in data:
        if x == " ":
            continue
        result.append(x)
    return result
