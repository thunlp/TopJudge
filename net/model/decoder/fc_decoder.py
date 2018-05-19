import torch
import torch.nn as nn
import torch.nn.functional as F
import configparser
from net.loader import get_num_classes


class FCDecoder(nn.Module):
    def __init__(self, config, usegpu):
        super(FCDecoder, self).__init__()
        try:
            features = config.getint("net", "fc1_feature")
        except configparser.NoOptionError:
            features = config.getint("net", "hidden_size")

        self.outfc = []
        task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
        for x in task_name:
            self.outfc.append(nn.Linear(
                features, get_num_classes(x)
            ))

        self.midfc = []
        for x in task_name:
            self.midfc.append(nn.Linear(features, features))

        self.outfc = nn.ModuleList(self.outfc)
        self.midfc = nn.ModuleList(self.midfc)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, doc_len, config):
        fc_input = x
        outputs = []
        now_cnt = 0
        for fc in self.outfc:
            if config.getboolean("net", "more_fc"):
                outputs.append(fc(F.relu(self.midfc[now_cnt](fc_input))))
            else:
                outputs.append(fc(fc_input))
            now_cnt += 1

        return outputs
