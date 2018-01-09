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

from model import CNN_FINAL, train_file
from file_reader import init_dataset

train_dataset, test_dataset = init_dataset(config)

print("Building net...")

net = CNN_FINAL(config)
if torch.cuda.is_available() and usegpu:
    net = net.cuda()

print("Net building done.")

train_file(net, train_dataset, test_dataset, usegpu, config)






