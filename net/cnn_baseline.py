import configparser
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c')
parser.add_argument('--gpu', '-g')
args = parser.parse_args()

configFilePath = args.config
if configFilePath is None:
    print("python *.py\t--config/-c\tconfigfile")
usegpu = True
if args.use is None:
    print("python *.py\t--use/-u\tcpu/gpu")
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
momemtum = config.getfloat("train", "momentum")

output_time = config.getint("debug", "output_time")
test_time = config.getint("debug", "test_time")
num_class = config.get("data", "type_of_label").replace(" ", "").split(",")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.convs = []

        for a in range(config.getint("net", "min_gram"), config.getint("net", "max_gram") + 1):
            self.convs.append(nn.Conv2d(1, config.getint("net", "filters"), (a, config.getint("data", "vec_size"))))

        self.fc1 = nn.Linear(
            (config.getint("net", "max_gram") - config.getint("net", "min_gram") + 1) * config.getint("net", "filters"),
            config.getint("net", "fc1_feature"))
        self.outfc = []
        for x in num_class:
            self.outfc.append(nn.Linear(
                config.getint("net", "fc1_feature"), get_num_classes(x)
            ))

        self.softmax = nn.Softmax(dim=1)

        self.convs = nn.ModuleList(self.convs)
        self.outfc = nn.ModuleList(self.outfc)

    def forward(self, x):
        fc_input = []
        for conv in self.convs:
            fc_input.append(torch.max(conv(x), dim=2, keepdim=True)[0])

        # for x in fc_input:
        #    print(x)

        fc_input = torch.cat(fc_input, dim=1).view(-1, (
            config.getint("net", "max_gram") - config.getint("net", "min_gram") + 1) * config.getint("net", "filters"))

        fc1_out = F.relu(self.fc1(fc_input))
        output = []
        for fc in self.outfc:
            output.append(self.softmax(fc(fc1_out)))
            # output = self.softmax(self.fc2(fc1_out))

        return output


net = Net()
if torch.cuda.is_available() and usegpu:
    net = net.cuda()

criterion = nn.NLLLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momemtum)


def test():
    pass


total_loss = []

for epoch_num in range(0, epoch):
    running_loss = 0
    cnt = 0
    for idx, data in enumerate(train_data_loader):
        cnt += 1
        input, label = data
        if torch.cuda.is_available() and usegpu:
            input, label = Variable(input.cuda()), Variable(label.cuda())
        else:
            input, label = Variable(input), Variable(label)

        optimizer.zero_grad()

        outputs = net.forward(input)
        loss = 0
        for a in range(0, len(num_class)):
            loss = loss + criterion(outputs[a], label[a])
        # loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]

        if cnt % output_time == 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch_num + 1, idx + 1, running_loss / output_time))
            total_loss.append(running_loss / output_time)
            running_loss = 0.0

        if cnt % test_time == 0:
            test()
