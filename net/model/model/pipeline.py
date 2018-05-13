import torch
import torch.nn as nn
from torch.autograd import Variable

from net.model.encoder import CNNEncoder
from net.loader import get_num_classes


def one_hot(indexes, nr_classes):
    zeros = Variable(torch.zeros(indexes.size() + (nr_classes,), out=indexes.data.new()))
    ones = torch.ones_like(indexes)
    zeros.scatter_(-1, indexes.unsqueeze(-1), ones.unsqueeze(-1))
    return zeros.float()


class Pipeline(nn.Module):
    def __init__(self, config, usegpu):
        super(Pipeline, self).__init__()

        self.encoder_list = []
        self.task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
        self.features = config.getint("net", "fc1_feature")
        for a in range(0, len(self.task_name)):
            self.encoder_list.append(CNNEncoder(config, usegpu))
        self.encoder_list = nn.ModuleList(self.encoder_list)

        self.out_fc = []
        for a in range(0, len(self.task_name)):
            self.out_fc.append(nn.Linear(self.features, get_num_classes(self.task_name[a])))
        self.out_fc = nn.ModuleList(self.out_fc)

        self.mix_fc = []
        for a in range(0, len(self.task_name)):
            mix_fc = []
            for b in range(0, len(self.task_name)):
                mix_fc.append(nn.Linear(get_num_classes(self.task_name[a]), self.features))
            mix_fc = nn.ModuleList(mix_fc)
            self.mix_fc.append(mix_fc)
        self.mix_fc = nn.ModuleList(self.mix_fc)

        self.combine_fc = []
        for a in range(0, len(self.task_name)):
            self.combine_fc.append(nn.Linear(self.features, self.features))
        self.combine_fc = nn.ModuleList(self.combine_fc)

        self.dropout = nn.Dropout(config.getfloat("train", "dropout"))
        self.softmax = nn.Softmax()

    def init_hidden(self, config, usegpu):
        pass

    def forward(self, x, doc_len, config, label):
        label_list = []
        accumulate = 0
        for a in range(0, len(self.task_name)):
            num = get_num_classes(self.task_name[a])
            label_list.append(label[:, accumulate:accumulate + num].float())
            accumulate += num

        outputs = []
        format_outputs = []
        for a in range(0, len(self.task_name)):
            document_embedding = self.combine_fc[a](self.encoder_list[a].forward(x, doc_len, config))
            for b in range(0, a):
                if self.training:
                    document_embedding = document_embedding + self.mix_fc[b][a](label_list[b])
                else:
                    document_embedding = document_embedding + self.mix_fc[b][a](format_outputs[b])
            output = self.out_fc[a](document_embedding)
            outputs.append(output)
            output = torch.max(output, dim=1)[1]
            output = one_hot(output, get_num_classes(self.task_name[a]))
            format_outputs.append(output)

        return outputs
