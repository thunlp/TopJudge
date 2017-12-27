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
import torch.optim as optim

from data_fetcher import init_loader, get_num_classes

train_data_loader, test_data_loader = init_loader(config)

epoch = config.getint("train", "epoch")
batch_size = config.getint("data", "batch_size")
learning_rate = config.getfloat("train", "learning_rate")
print(learning_rate)
momemtum = config.getfloat("train", "momentum")

output_time = config.getint("debug", "output_time")
test_time = config.getint("debug", "test_time")
task_name = config.get("data", "type_of_label").replace(" ", "").split(",")

print("Building net...")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.data_size = config.getint("data", "vec_size")
        self.hidden_dim = config.getint("net", "hidden_size")

        self.lstm = nn.LSTM(self.data_size, self.hidden_dim)

        self.outfc = []
        for x in task_name:
            self.outfc.append(nn.Linear(
                self.hidden_dim*config.getint("data","pad_length"), get_num_classes(x)
            ))
        self.outfc = nn.ModuleList(self.outfc)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if torch.cuda.is_available() and usegpu:
            return (torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim).cuda()),
                    torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim).cuda()))
        else:
            return (torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                    torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, x):
        x = x.view(config.getint("data","batch_size"),config.getint("data","pad_length"),config.getint("data","vec_size"))
        #print(x)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        lstm_out = lstm_out.view(config.getint("data","batch_size"),-1)

        outputs = []
        for fc in self.outfc:
            outputs.append(fc(lstm_out))
            # output = self.softmax(self.fc2(fc1_out))
        #print(outputs)

        return outputs


net = Net()
if torch.cuda.is_available() and usegpu:
    net = net.cuda()
print("Net building done.")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momemtum)


def calc_accuracy(outputs, labels):
    # print(labels)
    # print(outputs[0])
    # print(outputs[1])
    # print(outputs[2])
    # print(outputs.max(dim=1)[1])
    # print(outputs.max(dim=1)[1]-labels)
    return ((outputs.max(dim=1)[1].eq(labels)).sum(), len(labels))


def test():
    running_acc = []
    for a in range(0, len(task_name)):
        running_acc.append((0, 0))
    for idx, data in enumerate(test_data_loader):

        net.hidden = net.init_hidden()
        inputs, labels = data

        if torch.cuda.is_available() and usegpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = net.forward(inputs)
        for a in range(0, len(task_name)):
            x, y = running_acc[a]
            r, z = calc_accuracy(outputs[a], labels.transpose(0, 1)[a])
            running_acc[a] = (x + r, y + z)
        # loss = criterion(outputs, label)
        # print(loss.data[0])
        optimizer.step()

    print('Test accuracy:')
    # print(running_acc)
    for a in range(0, len(task_name)):
        # print(running_acc[a][0].data[0],running_acc[a][1])
        print("%s\t%.3f\t%d\t%d" % (
            task_name[a], running_acc[a][0].data[0] / running_acc[a][1], running_acc[a][0].data[0],
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
    for idx, data in enumerate(train_data_loader):
        net.hidden = net.init_hidden()
        cnt += 1
        inputs, labels = data
        # print(inputs)
        # print(net.fc1)
        # gg
        # print(inputs)
        # print(labels)
        if torch.cuda.is_available() and usegpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        optimizer.zero_grad()

        outputs = net.forward(inputs)
        # print(outputs)
        loss = 0
        for a in range(0, len(task_name)):
            loss = loss + criterion(outputs[a], labels.transpose(0, 1)[a])
            x, y = running_acc[a]
            r, z = calc_accuracy(outputs[a], labels.transpose(0, 1)[a])
            running_acc[a] = (x + r, y + z)
        # loss = criterion(outputs, label)
        # print(loss.data[0])
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]

        if cnt % output_time == 0:
            print('[%d, %5d, %5d] loss: %.3f' %
                  (epoch_num + 1, cnt, idx + 1, running_loss / output_time))
            print('accuracy:')
            # print(running_acc)
            for a in range(0, len(task_name)):
                # print(running_acc[a][0].data[0],running_acc[a][1])
                print("%s\t%.3f\t%d\t%d" % (
                    task_name[a], running_acc[a][0].data[0] / running_acc[a][1], running_acc[a][0].data[0],
                    running_acc[a][1]))
            print("")
            total_loss.append(running_loss / output_time)
            running_loss = 0.0
            for a in range(0, len(task_name)):
                running_acc[a] = (0, 0)

        if cnt % test_time == 0:
            test()

print("Training done")
