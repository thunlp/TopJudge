import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import os
import configparser
import argparse

from utils import calc_accuracy, gen_result, get_num_classes, generate_graph
import pdb


def test_file(net, test_dataset, usegpu, config, epoch):
    net.eval()
    running_acc = []
    task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
    if not (os.path.exists(config.get("train", "test_path"))):
        os.makedirs(config.get("train", "test_path"))
    test_result_path = os.path.join(config.get("train", "test_path"), str(epoch))
    for a in range(0, len(task_name)):
        running_acc.append([])
        for b in range(0, get_num_classes(task_name[a])):
            running_acc[a].append({"TP": 0, "FP": 0, "FN": 0})
            running_acc[a][-1]["list"] = []
            for c in range(0, get_num_classes(task_name[a])):
                running_acc[a][-1]["list"].append(0)

    while True:
        data = test_dataset.fetch_data(config)
        if data is None:
            break

        inputs, inputs_art, doc_len, labels = data

        net.init_hidden(config, usegpu)

        if torch.cuda.is_available() and usegpu:
            inputs, inputs_art, doc_len, labels = Variable(inputs.cuda()), Variable(inputs_art.cuda()), Variable(doc_len.cuda()), Variable(labels.cuda())
        else:
            inputs, inputs_art, doc_len, labels = Variable(inputs), Variable(inputs_art), Variable(doc_len), Variable(labels)

        outputs = net.forward(inputs, inputs_art, doc_len, config)
        for a in range(0, len(task_name)):
            running_acc[a] = calc_accuracy(outputs[a], labels.transpose(0, 1)[a], running_acc[a])

    net.train()

    print('Test result:')
    for a in range(0, len(task_name)):
        print("%s result:" % task_name[a])
        try:
            gen_result(running_acc[a], True, file_path=test_result_path + "-" + task_name[a])
        except Exception as e:
            pass
    print("")


def train_file(net, train_dataset, test_dataset, usegpu, config):
    epoch = config.getint("train", "epoch")
    batch_size = config.getint("data", "batch_size")
    learning_rate = config.getfloat("train", "learning_rate")
    momemtum = config.getfloat("train", "momentum")
    shuffle = config.getboolean("data", "shuffle")

    output_time = config.getint("debug", "output_time")
    test_time = config.getint("debug", "test_time")
    task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
    optimizer_type = config.get("train", "optimizer")

    model_path = config.get("train", "model_path")

    criterion = nn.CrossEntropyLoss()
    if optimizer_type == "adam":
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-3)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momemtum)
    else:
        gg

    total_loss = []
    first = True

    print("Training begin")
    for epoch_num in range(0, epoch):
        running_loss = 0
        running_acc = []
        for a in range(0, len(task_name)):
            running_acc.append([])
            for b in range(0, get_num_classes(task_name[a])):
                running_acc[a].append({"TP": 0, "FP": 0, "FN": 0})
                running_acc[a][-1]["list"] = []
                for c in range(0, get_num_classes(task_name[a])):
                    running_acc[a][-1]["list"].append(0)

        cnt = 0
        idx = 0
        while True:
            data = train_dataset.fetch_data(config)
            if data is None:
                break
            idx += batch_size
            cnt += 1

            inputs, inputs_art, doc_len, labels = data
            if torch.cuda.is_available() and usegpu:
                inputs, inputs_art, doc_len, labels = Variable(inputs.cuda()), Variable(inputs_art.cuda()), Variable(doc_len.cuda()), Variable(labels.cuda())
            else:
                inputs, inputs_art, doc_len, labels = Variable(inputs), Variable(inputs_art), Variable(doc_len), Variable(labels)

            net.init_hidden(config, usegpu)
            optimizer.zero_grad()

            outputs = net.forward(inputs, inputs_art, doc_len, config)
            loss = 0
            for a in range(0, len(task_name)):
                # print(outputs[a])
                # print(labels.transpose(0, 1)[a])
                loss = loss + criterion(outputs[a], labels.transpose(0, 1)[a])
                running_acc[a] = calc_accuracy(outputs[a], labels.transpose(0, 1)[a], running_acc[a])

            if False:
                loss.backward(retain_graph=True)
                first = False
            else:
                loss.backward()
            # pdb.set_trace()

            optimizer.step()

            running_loss += loss.data[0]

            if cnt % output_time == 0:
                print('[%d, %5d, %5d] loss: %.3f' %
                      (epoch_num + 1, cnt, idx + 1, running_loss / output_time))
                for a in range(0, len(task_name)):
                    print("%s result:" % task_name[a])
                    gen_result(running_acc[a])
                print("")

                total_loss.append(running_loss / output_time)
                running_loss = 0.0
                running_acc = []
                for a in range(0, len(task_name)):
                    running_acc.append([])
                    for b in range(0, get_num_classes(task_name[a])):
                        running_acc[a].append({"TP": 0, "FP": 0, "FN": 0})
                        running_acc[a][-1]["list"] = []
                        for c in range(0, get_num_classes(task_name[a])):
                            running_acc[a][-1]["list"].append(0)

        test_file(net, test_dataset, usegpu, config, epoch_num + 1)
        if not (os.path.exists(model_path)):
            os.makedirs(model_path)
        torch.save(net.state_dict(), os.path.join(model_path, "model-%d.pkl" % (epoch_num + 1)))

    print("Training done")

    test_file(net, test_dataset, usegpu, config, 0)
    torch.save(net.state_dict(), os.path.join(model_path, "model-0.pkl"))
if __name__ == "__main__":
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

    from model import NN_fact_art
    from file_reader_gzp import init_dataset

    train_dataset, test_dataset = init_dataset(config)

    print("Building net...")

    net = NN_fact_art(config, usegpu)
    if torch.cuda.is_available() and usegpu:
        net = net.cuda()

    print("Net building done.")

    train_file(net, train_dataset, test_dataset, usegpu, config)