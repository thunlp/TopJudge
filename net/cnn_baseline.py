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
if args.gpu is None:
    usegpu = False
else:
    usegpu = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

config = configparser.RawConfigParser()
config.read(configFilePath)

import torch

from model import CNN, train
from data_fetcher import init_dataset

train_dataset, test_dataset = init_dataset(config)

print("Building net...")

net = CNN(config)
if torch.cuda.is_available() and usegpu:
    net = net.cuda()
print("Net building done.")

train(net, train_dataset, test_dataset, usegpu, config)
