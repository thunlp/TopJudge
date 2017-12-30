import torch


def get_num_classes(s):
    if s == "crit":
        return 50
    if s == "law":
        return gg
    if s == "time":
        return 12
    gg


def calc_accuracy(outputs, labels, res):
    for a in range(0, len(labels)):
        it_is = int(outputs[a].max(dim=0)[1].data.cpu().numpy())
        should_be = int(labels[a].data.cpu().numpy())
        if it_is == should_be:
            res[it_is]["TP"] += 1
        else:
            res[it_is]["FP"] += 1
            res[should_be]["FN"] += 1
    return res


def gen_result(res):
    precision = []
    recall = []
    f1 = []
    total = {}
    total["TP"] = 0
    total["FP"] = 0
    total["FN"] = 0
    for a in range(0, len(res)):
        total["TP"] += res[a]["TP"]
        total["FP"] += res[a]["FP"]
        total["FN"] += res[a]["FN"]
        if res[a]["TP"] + res[a]["FP"] != 0:
            precision.append(res[a]["TP"] / (res[a]["TP"] + res[a]["FP"]))
        else:
            precision.append(0)
        if res[a]["TP"] + res[a]["FN"] != 0:
            recall.append(res[a]["TP"] / (res[a]["TP"] + res[a]["FN"]))
        else:
            recall.append(0)
        if precision[a] + recall[a] != 0:
            f1.append(2 * precision[a] * recall[a] / (precision[a] + recall[a]))
        else:
            f1.append(0)

    # for a in range(0, len(res)):
    #    print("Class\t%d:\tprecesion\t%.3f\trecall\t%.3f\tf1\t%.3f" % (a, precesion[a], recall[a], f1[a]))

    print(total["TP"], total["FP"], total["FN"])
    micro_precision = total["TP"] / (total["TP"] + total["FP"])
    macro_precision = 0
    micro_recall = total["TP"] / (total["TP"] + total["FN"])
    macro_recall = 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    macro_f1 = 0
    for a in range(0, len(res)):
        macro_precision += precision[a]
        macro_recall += recall[a]
        macro_f1 += f1[a]

    macro_precision /= len(res)
    macro_recall /= len(res)
    macro_f1 /= len(res)

    print("Micro precisison\t%.3f" % micro_precision)
    print("Macro precisison\t%.3f" % macro_precision)
    print("Micro recall\t%.3f" % micro_recall)
    print("Macro recall\t%.3f" % macro_recall)
    print("Micro f1\t%.3f" % micro_f1)
    print("Macro f1\t%.3f" % macro_f1)
