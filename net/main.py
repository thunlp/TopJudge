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
from model import *
from file_reader import init_dataset

train_dataset, test_dataset = init_dataset(config)

print("Building net...")
net = None

model_name = config.get("net", "name")

if model_name == "CNN":
    net = CNN(config, usegpu)
elif model_name == "MULTI_LSTM":
    net = MULTI_LSTM(config, usegpu)
elif model_name == "CNN_FINAL":
    net = CNN_FINAL(config, usegpu)
elif model_name == "MULTI_LSTM_FINAL":
    net = MULTI_LSTM_FINAL(config, usegpu)
elif model_name == "ARTICLE":
    net = ARTICLE(config, usegpu)
else:
    gg
if torch.cuda.is_available() and usegpu:
    net = net.cuda()

print("Net building done.")

try:
    train_file(net, train_dataset, test_dataset, usegpu, config)
except Exception as e:
    print(e)
    for x in train_dataset.read_process:
        x.terminate()
        print(x, x.is_alive())
        x.join()
        print(x, x.is_alive())
    for x in test_dataset.read_process:
        x.terminate()
        print(x, x.is_alive())
        x.join()
        print(x, x.is_alive())
