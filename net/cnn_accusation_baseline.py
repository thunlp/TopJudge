import configparser

configFilePath = "config.py"
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
            self.convs.append(nn.Conv2d(1, config.getint("net", "filters"), (a, config.getint("word", "vec_size"))))

        self.fc1 = nn.Linear(
            (config.getint("net", "max_gram") - config.getint("net", "min_gram") + 1) * config.getint("net", "filters"),
            config.getint("fc1_feature"))
        self.fc2 = nn.Linear(
            config.getint("fc1_feature"), config.getint("num_classes")
        )

        self.softmax = nn.LogSoftmax(dim=1)

        self.convs = nn.ModuleList(self.convs)

    def forward(self, x):
        fc_input = None
        for conv in self.convs:
            if fc_input is None:
                fc_input = torch.max(conv(x), dim=2, keepdim=True)
            else:
                fc_input = torch.cat((fc_input, torch.max(conv(x), dim=2, keepdim=True)), 1)

        fc1_out = F.relu(self.fc1(fc_input))
        output = self.softmax(self.fc2(fc1_out))

        return output


import math
import time
import torch.optim as optim

epoch = config.getint("train", "epoch")
iteration = config.getint("train", "iteration")
bacth_size = config.getint("train", "batch_size")
learning_rate = config.getfloat("train", "learning_rate")
momemtum = config.getfloat("train", "momentum")

net = Net()

criterion = nn.NLLLoss()
optimize = optim.SGD(net.parameters(), lr=learning_rate, momentum=momemtum)

for epoch_num in range(0, epoch):
    for iteration_num in (0, iteration):
