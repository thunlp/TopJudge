import configparser

configFilePath = "config.py"
config = configparser.RawConfigParser()
config.read(configFilePath)

import torch
import torch.nn as nn
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.convs = []

        for a in range(config.getint("net", "min_gram"), config.getint("net", "max_gram") + 1):
            self.convs.append(nn.Conv2d(1, config.getint("net", "filters"), (a, config.getint("word", "vec_size"))))

        self.fc1 = nn.Linear(
            (config.getint("net", "max_gram") - config.getint("net", "min_gram") + 1) * config.getint("filters"),
            config.getint("fc1_feature"))

        self.convs = nn.ModuleList(self.convs)

    def forward(self, x):
        mid_value = []
        for conv in self.convs:
            mid_value.append(conv(x))

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


if __name__ == "__main__":
