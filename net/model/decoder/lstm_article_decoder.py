import torch
import torch.nn as nn
import torch.nn.functional as F

from net.model.encoder import ArticleEncoder
from net.model.layer.attention import Attention
from net.utils import  generate_graph
from net.loader import get_num_classes


class LSTMArticleDecoder(nn.Module):
    def __init__(self, config, usegpu):
        super(LSTMArticleDecoder, self).__init__()
        self.feature_len = config.getint("net", "hidden_size")

        features = config.getint("net", "hidden_size")
        self.hidden_dim = features
        self.outfc = []
        task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
        for x in task_name:
            self.outfc.append(nn.Linear(
                features, get_num_classes(x)
            ))

        self.midfc = []
        for x in task_name:
            self.midfc.append(nn.Linear(features, features))

        self.cell_list = [None]
        for x in task_name:
            self.cell_list.append(nn.LSTMCell(config.getint("net", "hidden_size"), config.getint("net", "hidden_size")))

        self.hidden_state_fc_list = []
        for a in range(0, len(task_name) + 1):
            arr = []
            for b in range(0, len(task_name) + 1):
                arr.append(nn.Linear(features, features))
            arr = nn.ModuleList(arr)
            self.hidden_state_fc_list.append(arr)

        self.cell_state_fc_list = []
        for a in range(0, len(task_name) + 1):
            arr = []
            for b in range(0, len(task_name) + 1):
                arr.append(nn.Linear(features, features))
            arr = nn.ModuleList(arr)
            self.cell_state_fc_list.append(arr)

        self.attention = Attention(config)
        self.outfc = nn.ModuleList(self.outfc)
        self.midfc = nn.ModuleList(self.midfc)
        self.cell_list = nn.ModuleList(self.cell_list)
        self.hidden_state_fc_list = nn.ModuleList(self.hidden_state_fc_list)
        self.cell_state_fc_list = nn.ModuleList(self.cell_state_fc_list)
        self.sigmoid = nn.Sigmoid()

        self.article_encoder = ArticleEncoder(config, usegpu)
        self.article_fc_list = []
        for a in range(0, len(task_name) + 1):
            self.article_fc_list.append(nn.Linear(features, features))
        self.article_fc_list = nn.ModuleList(self.article_fc_list)

    def init_hidden(self, config, usegpu):
        self.article_encoder.init_hidden(config, usegpu)
        self.hidden_list = []
        task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
        for a in range(0, len(task_name) + 1):
            if torch.cuda.is_available() and usegpu:
                self.hidden_list.append((
                    torch.autograd.Variable(
                        torch.zeros(config.getint("data", "batch_size"), self.hidden_dim).cuda()),
                    torch.autograd.Variable(
                        torch.zeros(config.getint("data", "batch_size"), self.hidden_dim).cuda())))
            else:
                self.hidden_list.append((
                    torch.autograd.Variable(torch.zeros(config.getint("data", "batch_size"), self.hidden_dim)),
                    torch.autograd.Variable(torch.zeros(config.getint("data", "batch_size"), self.hidden_dim))))

    def forward(self, x, doc_len, config, attention):
        fc_input = x
        outputs = []
        task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
        graph = generate_graph(config)

        first = []
        for a in range(0, len(task_name) + 1):
            first.append(True)
        for a in range(1, len(task_name) + 1):
            h, c = self.cell_list[a](fc_input, self.hidden_list[a])
            for b in range(1, len(task_name) + 1):
                if graph[a][b]:
                    hp, cp = self.hidden_list[b]
                    if first[b]:
                        first[b] = False
                        hp, cp = h, c
                    else:
                        hp = hp + self.hidden_state_fc_list[a][b](h)
                        cp = cp + self.cell_state_fc_list[a][b](c)
                    self.hidden_list[b] = (hp, cp)
            # self.hidden_list[a] = h, c
            if config.getboolean("net", "attention"):
                h = self.attention(h, attention)
            if config.getboolean("net", "more_fc"):
                outputs.append(
                    self.outfc[a - 1](F.relu(self.midfc[a - 1](h))).view(config.getint("data", "batch_size"), -1))
            else:
                outputs.append(self.outfc[a - 1](h).view(config.getint("data", "batch_size"), -1))

            if a == 1:
                article_embedding = self.article_encoder(outputs[a - 1], doc_len, config)
                for b in range(a + 1, len(task_name) + 1):
                    hp, cp = self.hidden_list[b]
                    if first[b]:
                        first[b] = False
                        hp, cp = article_embedding, c
                    else:
                        hp = hp + self.article_fc_list[b](article_embedding)

                    self.hidden_list[b] = (hp, cp)

        return outputs
