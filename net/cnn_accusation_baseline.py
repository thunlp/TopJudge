import configparser
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c')
args = parser.parse_args()

configFilePath = args.config
if configFilePath is None:
    print("python *.py\t--config/-c\tconfigfile")
config = configparser.RawConfigParser()
config.read(configFilePath)

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.convs = []

        for a in range(config.getint("net", "min_gram"), config.getint("net", "max_gram") + 1):
            self.convs.append(nn.Conv2d(1, config.getint("net", "filters"), (a, config.getint("data", "vec_size"))))

        self.fc1 = nn.Linear(
            (config.getint("net", "max_gram") - config.getint("net", "min_gram") + 1) * config.getint("net", "filters"),
            config.getint("net", "fc1_feature"))
        self.fc2 = nn.Linear(
            config.getint("net", "fc1_feature"), config.getint("data", "num_classes")
        )

        self.softmax = nn.LogSoftmax(dim=1)

        self.convs = nn.ModuleList(self.convs)

    def forward(self, x):
        fc_input = []
        for conv in self.convs:
            fc_input.append(torch.max(conv(x), dim=2, keepdim=True)[0])

        # for x in fc_input:
        #    print(x)

        fc_input = torch.cat(fc_input, dim=1).view(-1, (
            config.getint("net", "max_gram") - config.getint("net", "min_gram") + 1) * config.getint("net", "filters"))

        fc1_out = F.relu(self.fc1(fc_input))
        output = self.softmax(self.fc2(fc1_out))

        return output


import math
import time
import torch.optim as optim

epoch = config.getint("train", "epoch")
iteration = config.getint("train", "iteration")
batch_size = config.getint("train", "batch_size")
learning_rate = config.getfloat("train", "learning_rate")
momemtum = config.getfloat("train", "momentum")

output_time = config.getint("debug", "output_time")
test_time = config.getint("debug", "test_time")

net = Net()
if torch.cuda.is_available():
    net = net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momemtum)

from data_fetcher import init_loader

train_data_loader = init_loader(config)

for epoch_num in range(0, epoch):
    running_loss = 0
    cnt = 0
    for idx, data in enumerate(train_data_loader):
        cnt += 1
        input, label = data
        if torch.cuda.is_available():
            input, label = Variable(input.cuda()), Variable(label.cuda())
        else:
            input, label = Variable(input), Variable(label)

        optimizer.zero_grad()

        outputs = net.forward(input)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]

        if idx % output_time == output_time - 1:
            print('[%d, %5d] loss: %.3f' %
                  (epoch_num + 1, idx + 1, running_loss / output_time))
            running_loss = 0.0
