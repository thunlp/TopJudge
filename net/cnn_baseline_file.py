import configparser
import argparse
import os
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c')
parser.add_argument('--gpu', '-g')
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

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import time
from torch.utils.data import DataLoader
import torch.optim as optim

from data_fetcher import init_dataset, get_num_classes
from utils import calc_accuracy, gen_result


train_dataset, test_dataset = init_dataset(config)

epoch = config.getint("train", "epoch")
batch_size = config.getint("data", "batch_size")
learning_rate = config.getfloat("train", "learning_rate")
print(learning_rate)
momemtum = config.getfloat("train", "momentum")
shuffle = config.getboolean("data", "shuffle")

output_time = config.getint("debug", "output_time")
test_time = config.getint("debug", "test_time")
task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
optimizer_type = config.get("train", "optimizer")

print("Building net...")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.convs = []

        for a in range(config.getint("net", "min_gram"), config.getint("net", "max_gram") + 1):
            self.convs.append(nn.Conv2d(1, config.getint("net", "filters"), (a, config.getint("data", "vec_size"))))

        features = (config.getint("net", "max_gram") - config.getint("net", "min_gram") + 1) * config.getint("net",
                                                                                                             "filters")
        self.outfc = []
        for x in task_name:
            self.outfc.append(nn.Linear(
                features, get_num_classes(x)
            ))

        self.convs = nn.ModuleList(self.convs)
        self.outfc = nn.ModuleList(self.outfc)

    def forward(self, x):
        fc_input = []
        for conv in self.convs:
            fc_input.append(torch.max(conv(x), dim=2, keepdim=True)[0])


        features = (config.getint("net", "max_gram") - config.getint("net", "min_gram") + 1) * config.getint("net",
                                                                                                             "filters")

        fc_input = torch.cat(fc_input, dim=1).view(-1, features)

        outputs = []
        for fc in self.outfc:
            outputs.append(fc(fc_input))

        return outputs


net = Net()
if torch.cuda.is_available() and usegpu:
    net = net.cuda()
print("Net building done.")

criterion = nn.CrossEntropyLoss()
if optimizer_type == "adam":
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
elif optimizer_type == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momemtum)
else:
    gg



def test():
    running_acc = []
    for a in range(0, len(task_name)):
        running_acc.append([])
        for b in range(0, get_num_classes(task_name[a])):
            running_acc[a].append({"TP": 0, "FP": 0, "FN": 0})

    while True:
        data = test_dataset.fetch_data(config)
        if data is None:
            break

        inputs, doc_len, labels = data

        if torch.cuda.is_available() and usegpu:
            inputs, doc_len, labels = Variable(inputs.cuda()), Variable(doc_len.cuda()), Variable(labels.cuda())
        else:
            inputs, doc_len, labels = Variable(inputs), Variable(doc_len), Variable(labels)

        outputs = net.forward(inputs)
        for a in range(0, len(task_name)):
            x, y = running_acc[a]
            r, z = calc_accuracy(outputs[a], labels.transpose(0, 1)[a])
            running_acc[a] = (x + r, y + z)
        break
    if running_acc[a][1] == 0:
        return

    print('Test accuracy:')
    for a in range(0, len(task_name)):
        print("%s\t%.3f\t%d\t%d" % (
            task_name[a], running_acc[a][0] / running_acc[a][1], running_acc[a][0],
            running_acc[a][1]))
    print("")


total_loss = []

print("Training begin")

for epoch_num in range(0, epoch):
    running_loss = 0
    running_acc = []
    for a in range(0, len(task_name)):
        running_acc.append((0, 0))
    cnt = 0
    idx = 0
    while True:
        data = train_dataset.fetch_data(config)
        if data is None:
            break
        idx += batch_size
        cnt += 1

        inputs, doc_len, labels = data
        if torch.cuda.is_available() and usegpu:
            inputs, doc_len, labels = Variable(inputs.cuda()), Variable(doc_len.cuda()), Variable(labels.cuda())
        else:
            inputs, doc_len, labels = Variable(inputs), Variable(doc_len), Variable(labels)

        optimizer.zero_grad()

        outputs = net.forward(inputs)
        loss = 0
        for a in range(0, len(task_name)):
            loss = loss + criterion(outputs[a], labels.transpose(0, 1)[a])
            x, y = running_acc[a]
            r, z = calc_accuracy(outputs[a], labels.transpose(0, 1)[a])
            running_acc[a] = (x + r, y + z)
        # pdb.set_trace()
        loss.backward()
        optimizer.step()
        # pdb.set_trace()

        running_loss += loss.data[0]

        if cnt % output_time == 0:
            print('[%d, %5d, %5d] loss: %.3f' %
                  (epoch_num + 1, cnt, idx + 1, running_loss / output_time))
            print('accuracy:')
            # print(running_acc)
            for a in range(0, len(task_name)):
                print("%s\t%.3f\t%d\t%d" % (
                    task_name[a] + "accuracy", running_acc[a][0] / running_acc[a][1], running_acc[a][0],
                    running_acc[a][1]))
            print("")
            total_loss.append(running_loss / output_time)
            running_loss = 0.0
            for a in range(0, len(task_name)):
                running_acc[a] = (0, 0)

        if cnt % test_time == 0:
            test()

print("Training done")

test()
