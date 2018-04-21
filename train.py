import configparser
import argparse
import os
import pdb
import torch

from net.model import *
from net.file_reader import init_dataset
from net.work import train_file
from net.utils import print_info

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

train_dataset, test_dataset = init_dataset(config)

print_info("Building net...")
net = None

model_name = config.get("net", "name")

match_list = {
    "CNN": CNN,
    "MultiLSTM": MultiLSTM,
    "CNNSeq": CNNSeq,
    "MultiLSTMSeq": MultiLSTMSeq,
    "Article": Article,
    "LSTM": LSTM
}

if model_name in match_list.keys():
    net = match_list[model_name](config, usegpu)
else:
    gg

try:
    net.load_state_dict(
        torch.load(
            os.path.join(config.get("train", "model_path"), "model-" + config.get("train", "pre_train") + ".pkl")))
except Exception as e:
    pass

if torch.cuda.is_available() and usegpu:
    net = net.cuda()

print_info("Net building done.")

train_file(net, train_dataset, test_dataset, usegpu, config)

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
