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
        # x = x.transpose(0,1)
        x = x.view(config.getint("data", "batch_size"), config.getint("data", "pad_length"),
                   config.getint("data", "vec_size"))
        # print(x)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        # print(lstm_out)
        # gg
        # lstm_out = lstm_out.transpose(0,1)
        # print(lstm_out)
        # lstm_out = lstm_out.permute(1,0,2)
        # print(doc_len)
        outv = []
        for a in range(0, len(doc_len)):
            outv.append(lstm_out[a][doc_len[a] - 1])
        lstm_out = torch.cat(outv)
        # lstm_out = lstm_out[:,-1]#lstm_out.view(config.getint("data", "batch_size"), -1)

        outputs = []
        for fc in self.outfc:
            outputs.append(fc(lstm_out))
            # output = self.softmax(self.fc2(fc1_out))
        # print(outputs)

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


def calc_accuracy(outputs, labels):
    v1 = int((outputs.max(dim=1)[1].eq(labels)).sum().data.cpu().numpy())
    v2 = 0
    for a in range(0, len(labels)):
        nowl = outputs[a].max(dim=0)[1]
        v2 += int(torch.eq(nowl, labels[a]).data.cpu().numpy())

        # if torch.eq(nowl,labels[a]) == 1:
        #    v2 += 1
    v3 = len(labels)
    if v1 != v2:
        print(outputs.max(dim=1))
        print(labels)
        gg
    return (v2, v3)


def test():
    running_acc = []
    for a in range(0, len(task_name)):
        running_acc.append((0, 0))
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)
    for idx, data in enumerate(test_data_loader):

        net.hidden = net.init_hidden()
        inputs, doc_len, labels = data

        if torch.cuda.is_available() and usegpu:
            inputs, doc_len, labels = Variable(inputs.cuda()), Variable(doc_len.cuda()), Variable(labels.cuda())
        else:
            inputs, doc_len, labels = Variable(inputs), Variable(doc_len), Variable(labels)

        outputs = net.forward(inputs, doc_len)
        for a in range(0, len(task_name)):
            x, y = running_acc[a]
            r, z = calc_accuracy(outputs[a], labels.transpose(0, 1)[a])
            running_acc[a] = (x + r, y + z)

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
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=4)
    for idx, data in enumerate(train_data_loader):
        net.hidden = net.init_hidden()
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
            x, y = running_acc[a]
            r, z = calc_accuracy(outputs[a], labels.transpose(0, 1)[a])
            running_acc[a] = (x + r, y + z)

        loss.backward()
        optimizer.step()

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
