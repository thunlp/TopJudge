import configparser
import argparse
import os
import pdb
import json
from utils import calc_accuracy, gen_result, get_num_classes, generate_graph

import torch
from torch.autograd import Variable

from model import CNN
from file_reader import init_dataset


def test_file(net, test_dataset, usegpu, config, save_path):
    net.eval()
    running_acc = []
    task_name = config.get("data", "type_of_label").replace(" ", "").split(",")

    f = open(save_path, "w")

    for a in range(0, len(task_name)):
        running_acc.append([])
        for b in range(0, get_num_classes(task_name[a])):
            running_acc[a].append({"TP": 0, "FP": 0, "FN": 0})
            running_acc[a][-1]["list"] = []
            for c in range(0, get_num_classes(task_name[a])):
                running_acc[a][-1]["list"].append(0)

    while True:
        data = test_dataset.fetch_data(config)
        # print(len(data))
        if data is None:
            break

        inputs, doc_len, labels = data

        net.init_hidden(config, usegpu)

        if torch.cuda.is_available() and usegpu:
            inputs, doc_len, labels = Variable(inputs.cuda()), Variable(doc_len.cuda()), Variable(labels.cuda())
        else:
            inputs, doc_len, labels = Variable(inputs), Variable(doc_len), Variable(labels)

        outputs = net.forward(inputs, doc_len, config)
        out_label = []
        for a in range(0, len(task_name)):
            running_acc[a], la_tmp = calc_accuracy(outputs[a], labels.transpose(0, 1)[a], running_acc[a])
            out_label.append(la_tmp)

        for a in range(len(labels)):
            for b in range(len(task_name)):
                f.write(str(int(out_label[b][a]))+"\t")
            f.write("\n")
        f.flush()

    net.train()

    print('Test result:')
    for a in range(0, len(task_name)):
        print("%s result:" % task_name[a])
        try:
            gen_result(running_acc[a], False)
        except Exception as e:
            pass
    print("")
    


parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c')
parser.add_argument('--modelfile', '-mf')
parser.add_argument('--model', '-m')
parser.add_argument('--gpu', '-g')
parser.add_argument('--save', '-s')
args = parser.parse_args()

configFilePath = args.config
if configFilePath is None:
    print("python *.py\t--config/-c\tconfigfile")
usegpu = True
# if args.use is None:
#    print("python *.py\t--use/-u\tcpu/gpu")
if args.gpu is None:
    usegpu = False
else:
    usegpu = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

config = configparser.RawConfigParser()
config.read(configFilePath)

_, test_dataset = init_dataset(config)

print("Building net...")

net = None

if(args.model == "CNN"):
    net = CNN(config)
else:
    gg

net.load_state_dict(torch.load(args.modelfile))

if torch.cuda.is_available() and usegpu:
    net = net.cuda()

print("Net building done.")

test_file(net, test_dataset, usegpu, config, args.save)