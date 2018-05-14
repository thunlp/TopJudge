import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from net.model.loss import cross_entropy_loss, one_cross_entropy_loss, log_regression
from net.utils import calc_accuracy, gen_result, print_info
from net.loader import get_num_classes
from net.model.model import Pipeline, NNFactArtSeq, NNFactArt


def test_file(net, test_dataset, usegpu, config, epoch):
    net.eval()
    running_acc = []
    task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
    task_loss_type = config.get("data", "type_of_loss").replace(" ", "").split(",")
    test_result_path = os.path.join(config.get("output", "test_path"), config.get("output", "model_name"))
    if not (os.path.exists(test_result_path)):
        os.makedirs(test_result_path)
    test_result_path = os.path.join(config.get("output", "test_path"), config.get("output", "model_name"), str(epoch))
    for a in range(0, len(task_name)):
        running_acc.append([])
        for b in range(0, get_num_classes(task_name[a])):
            running_acc[a].append({"TP": 0, "FP": 0, "FN": 0, "TN": 0})

    while True:
        data = test_dataset.fetch_data(config)
        if data is None:
            break

        inputs, doc_len, labels = data[0]
        content = data[1]

        net.init_hidden(config, usegpu)

        if torch.cuda.is_available() and usegpu:
            inputs, doc_len, labels = Variable(inputs.cuda()), Variable(doc_len.cuda()), Variable(labels.cuda())
        else:
            inputs, doc_len, labels = Variable(inputs), Variable(doc_len), Variable(labels)

        if isinstance(net, Pipeline):
            outputs = net.forward(inputs, doc_len, config, labels)
        elif isinstance(net, NNFactArtSeq) or isinstance(net, NNFactArt):
            outputs = net.forward(inputs, doc_len, config, content)
        else:
            outputs = net.forward(inputs, doc_len, config)

        reals = []
        accumulate = 0
        for a in range(0, len(task_name)):
            num_class = get_num_classes(task_name[a])
            reals.append(labels[:, accumulate:accumulate + num_class])
            accumulate += num_class

        labels = reals
        for a in range(0, len(task_name)):
            running_acc[a] = calc_accuracy(outputs[a], labels[a], task_loss_type[a], running_acc[a])

    net.train()

    print_info('Test result:')
    for a in range(0, len(task_name)):
        print("%s result:" % task_name[a])
        try:
            gen_result(running_acc[a], True, file_path=test_result_path + "-" + task_name[a], class_name=task_name[a])
        except Exception as e:
            pass
    print("")


def train_file(net, train_dataset, test_dataset, usegpu, config):
    epoch = config.getint("train", "epoch")
    batch_size = config.getint("data", "batch_size")
    learning_rate = config.getfloat("train", "learning_rate")
    momemtum = config.getfloat("train", "momentum")
    weight_decay = config.getfloat("train", "weight_decay")

    output_time = config.getint("output", "output_time")
    task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
    task_loss_type = config.get("data", "type_of_loss").replace(" ", "").split(",")
    optimizer_type = config.get("train", "optimizer")

    model_path = os.path.join(config.get("output", "model_path"), config.get("output", "model_name"))
    test_time = config.getint("output", "test_time")

    criterion = []
    for a in range(0, len(task_name)):
        if task_loss_type[a] == "multi_classification":
            criterion.append(cross_entropy_loss)
        elif task_loss_type[a] == "single_classification":
            criterion.append(one_cross_entropy_loss)
        elif task_loss_type[a] == "log_regression":
            criterion.append(log_regression)
        else:
            gg
    if optimizer_type == "adam":
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momemtum)
    else:
        gg

    total_loss = []
    first = True
    try:
        epps = config.get("train", "pre_train")
        epps = int(epps) - 1
    except Exception as e:
        epps = -1

    print_info("Training begin")
    for epoch_num in range(epps + 1, epoch):
        running_loss = 0
        running_acc = []
        total_acc = []
        for a in range(0, len(task_name)):
            running_acc.append([])
            total_acc.append([])
            for b in range(0, get_num_classes(task_name[a])):
                running_acc[a].append({"TP": 0, "FP": 0, "FN": 0, "TN": 0})
                total_acc[a].append({"TP": 0, "FP": 0, "FN": 0, "TN": 0})

        cnt = 0
        idx = 0
        while True:
            # print_info("One round begin, waiting for data...")
            data = train_dataset.fetch_data(config)
            if data is None:
                break
            idx += batch_size
            cnt += 1

            inputs, doc_len, labels = data[0]
            content = data[1]

            if torch.cuda.is_available() and usegpu:
                inputs, doc_len, labels = Variable(inputs.cuda()), Variable(doc_len.cuda()), Variable(labels.cuda())
            else:
                inputs, doc_len, labels = Variable(inputs), Variable(doc_len), Variable(labels)

            # print_info("Data fetch done, forwarding...")

            net.init_hidden(config, usegpu)
            optimizer.zero_grad()

            if isinstance(net, Pipeline):
                outputs = net.forward(inputs, doc_len, config, labels)
            elif isinstance(net, NNFactArtSeq) or isinstance(net, NNFactArt):
                outputs = net.forward(inputs, doc_len, config, content)
            else:
                outputs = net.forward(inputs, doc_len, config)

            reals = []
            accumulate = 0
            for a in range(0, len(task_name)):
                num_class = get_num_classes(task_name[a])
                reals.append(labels[:, accumulate:accumulate + num_class])
                accumulate += num_class

            labels = reals

            # print_info("Forward done, lossing...")
            # print(labels)
            # print(outputs)
            loss = 0
            for a in range(0, len(task_name)):
                loss = loss + criterion[a](outputs[a], labels[a].float())
                running_acc[a] = calc_accuracy(outputs[a], labels[a], task_loss_type[a], running_acc[a])

            # print_info("Loss done, backwarding...")

            loss.backward()
            # pdb.set_trace()

            optimizer.step()

            running_loss += loss.data[0]

            # print_info("One round done, next round")

            if cnt % output_time == 0:
                print_info("Current res:")
                print('[%d, %5d, %5d] loss: %.3f' %
                      (epoch_num + 1, cnt, idx, running_loss / output_time))
                for a in range(0, len(task_name)):
                    print("%s result:" % task_name[a])
                    gen_result(running_acc[a])
                print("")

                total_loss.append(running_loss / output_time)
                running_loss = 0.0
                for a in range(0, len(running_acc)):
                    for b in range(0, len(running_acc[a])):
                        total_acc[a][b]["TP"] += running_acc[a][b]["TP"]
                        total_acc[a][b]["FP"] += running_acc[a][b]["FP"]
                        total_acc[a][b]["FN"] += running_acc[a][b]["FN"]
                        total_acc[a][b]["TN"] += running_acc[a][b]["TN"]

                running_acc = []
                for a in range(0, len(task_name)):
                    running_acc.append([])
                    for b in range(0, get_num_classes(task_name[a])):
                        running_acc[a].append({"TP": 0, "FP": 0, "FN": 0, "TN": 0})

        if not (os.path.exists(model_path)):
            os.makedirs(model_path)
        torch.save(net.state_dict(), os.path.join(model_path, "model-%d.pkl" % (epoch_num + 1)))

        if (epoch_num + 1) % test_time == 0:
            test_file(net, test_dataset, usegpu, config, epoch_num + 1)

        for a in range(0, len(task_name)):
            gen_result(total_acc[a], True,
                       file_path=os.path.join(config.get("output", "test_path"), config.get("output", "model_name"),
                                              "total") + "-" + task_name[a],
                       class_name=task_name[a])

    print_info("Training done")

    test_file(net, test_dataset, usegpu, config, 0)
    torch.save(net.state_dict(), os.path.join(model_path, "model.pkl"))
