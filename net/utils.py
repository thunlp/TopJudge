import torch
from torch.autograd import Variable
import json
import thulac
import pdb

cutter = thulac.thulac(model_path=r"/data/zhx/thulac/models", seg_only=True, filt=False)


def get_data_list(d):
    return d.replace(" ", "").split(",")


def calc_accuracy(outputs, labels, res):
    if len(labels[0]) != len(outputs[0]):
        gg
    for a in range(0, len(labels)):
        for b in range(0, len(labels[0])):
            if outputs[a][b].data[0] < 0.5:
                output_is = 0
            else:
                output_is = 1
            if labels[a][b].data[0] < 0.5:
                label_is = 0
            else:
                label_is = 1

            if label_is == 1:
                if output_is == 1:
                    res[b]["TP"] += 1
                else:
                    res[b]["FN"] += 1
            else:
                if output_is == 1:
                    res[b]["FP"] += 1
                else:
                    res[b]["TN"] += 1
    return res


def gen_result(res, test=False, file_path=None):
    precision = []
    recall = []
    f1 = []
    total = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for a in range(0, len(res)):
        total["TP"] += res[a]["TP"]
        total["FP"] += res[a]["FP"]
        total["FN"] += res[a]["FN"]
        total["TN"] += res[a]["FN"]

        # According to https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure

        if res[a]["TP"] == 0:
            if res[a]["FP"] == 0 and res[a]["FN"] == 0:
                precision.append(1)
                recall.append(1)
                f1.append(1)
            else:
                precision.append(0)
                recall.append(0)
                f1.append(0)
        else:
            precision.append(res[a]["TP"] / (res[a]["TP"] + res[a]["FP"]))
            recall.append(res[a]["TP"] / (res[a]["TP"] + res[a]["FN"]))
            f1.append(2 * precision[-1] * recall[-1] / (precision[-1] + recall[-1]))

    if total["TP"] + total["FP"] == 0:
        micro_precision = 0
    else:
        micro_precision = total["TP"] / (total["TP"] + total["FP"])
    macro_precision = 0
    if total["TP"] + total["FN"] == 0:
        micro_recall = 0
    else:
        micro_recall = total["TP"] / (total["TP"] + total["FN"])
    macro_recall = 0
    if micro_precision + micro_recall == 0:
        micro_f1 = 0
    else:
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
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
        print(" \t", file=f)
        for a in range(0, len(res)):
            print("%d\t" % a, end='', file=f)
        print("", file=f)
        for a in range(0, len(res)):
            print("%d " % a, res[a], file=f)
            print("", file=f)
        f.close()

    print("")


def generate_graph(config):
    s = config.get("data", "graph")
    arr = s.replace("[", "").replace("]", "").split(",")
    graph = []
    n = 0
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
