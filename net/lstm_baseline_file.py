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

from file_reader import init_dataset, get_num_classes
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

        self.data_size = config.getint("data", "vec_size")
        self.hidden_dim = config.getint("net", "hidden_size")

        self.lstm = nn.LSTM(self.data_size, self.hidden_dim, batch_first=True)

        self.outfc = []
        for x in task_name:
            self.outfc.append(nn.Linear(
                self.hidden_dim, get_num_classes(x)
            ))
        self.outfc = nn.ModuleList(self.outfc)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if torch.cuda.is_available() and usegpu:
            return (
                torch.autograd.Variable(torch.zeros(1, config.getint("data", "batch_size"), self.hidden_dim).cuda()),
                torch.autograd.Variable(torch.zeros(1, config.getint("data", "batch_size"), self.hidden_dim).cuda()))
        else:
            return (torch.autograd.Variable(torch.zeros(1, config.getint("data", "batch_size"), self.hidden_dim)),
                    torch.autograd.Variable(torch.zeros(1, config.getint("data", "batch_size"), self.hidden_dim)))

    def forward(self, x, doc_len):

        x = x.view(config.getint("data", "batch_size"), config.getint("data", "pad_length"),
                   config.getint("data", "vec_size"))

        lstm_out, self.hidden = self.lstm(x, self.hidden)

        outv = []
        for a in range(0, len(doc_len)):
            outv.append(lstm_out[a][doc_len[a] - 1])
        lstm_out = torch.cat(outv)

        outputs = []
        for fc in self.outfc:
            outputs.append(fc(lstm_out))


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

        net.hidden = net.init_hidden()
        inputs, doc_len, labels = data

        if torch.cuda.is_available() and usegpu:
            inputs, doc_len, labels = Variable(inputs.cuda()), Variable(doc_len.cuda()), Variable(labels.cuda())
        else:
            inputs, doc_len, labels = Variable(inputs), Variable(doc_len), Variable(labels)

        outputs = net.forward(inputs, doc_len)
        for a in range(0, len(task_name)):
            running_acc[a] = calc_accuracy(outputs[a], labels.transpose(0, 1)[a], running_acc[a])

    print('Test result:')
    for a in range(0, len(task_name)):
        print("%s result:" % task_name[a])
        try:
            gen_result(running_acc[a])
        except Exception as e:
            pass
    print("")


total_loss = []

print("Training begin")

for epoch_num in range(0, epoch):
    running_loss = 0
    running_acc = []
    for a in range(0, len(task_name)):
        running_acc.append([])
        for b in range(0, get_num_classes(task_name[a])):
            running_acc[a].append({"TP": 0, "FP": 0, "FN": 0})

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

        outputs = net.forward(inputs, doc_len)
        # print(outputs)
        loss = 0
        for a in range(0, len(task_name)):
            loss = loss + criterion(outputs[a], labels.transpose(0, 1)[a])
            running_acc[a] = calc_accuracy(outputs[a], labels.transpose(0, 1)[a], running_acc[a])

        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]

        if cnt % output_time == 0:
            print('[%d, %5d, %5d] loss: %.3f' %
                  (epoch_num + 1, cnt, idx + 1, running_loss / output_time))
            for a in range(0, len(task_name)):
                print("%s result:" % task_name[a])
                gen_result(running_acc[a])
            print("")

            total_loss.append(running_loss / output_time)
            running_loss = 0.0
            running_acc = []
            for a in range(0, len(task_name)):
                 running_acc.append([])
                 for b in range(0, get_num_classes(task_name[a])):
                     running_acc[a].append({"TP": 0, "FP": 0, "FN": 0})


    test()

print("Training done")

test()
