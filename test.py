import argparse
import os
import torch

from net.model import get_model
from net.file_reader import init_dataset, init_transformer
from net.work import test_file
from net.parser import ConfigParser
from net.loader import init

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m')
parser.add_argument('--gpu', '-g')
parser.add_argument('--config', '-c')
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

config = ConfigParser(configFilePath)
config.config.set("train", "train_num_process", 0)

init(config)
init_transformer(config)
train_dataset, test_dataset = init_dataset(config)

print("Building net...")
model_name = config.get("net", "name")

net = get_model(model_name, config, usegpu)

print("Net building done.")

print("Loading model...")
net.load_state_dict(torch.load(args.model))
if torch.cuda.is_available() and usegpu:
    net = net.cuda()
print("Model loaded.")

print("Testing model...")
test_file(net, test_dataset, usegpu, config, 0)
print("Test done.")
