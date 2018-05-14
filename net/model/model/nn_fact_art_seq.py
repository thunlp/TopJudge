import torch
import torch.nn as nn
import torch.nn.functional as F

from net.utils import generate_graph
from net.loader import get_num_classes
from net.model.layer import AttentionTanH


class NNFactArtSeq(nn.Module):
    def __init__(self, config, usegpu):
        super(NNFactArtSeq, self).__init__()

        self.data_size = config.getint("data", "vec_size")
        self.hidden_dim = config.getint("net", "hidden_size")
        self.top_k = config.getint("data", "top_k")

        self.gru_sentence_f = nn.GRU(self.data_size, self.hidden_dim, batch_first=True)
        self.gru_document_f = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)

        self.gru_sentence_a = []
        self.gru_document_a = []
        for i in range(self.top_k):
            self.gru_sentence_a.append(nn.GRU(self.data_size, self.hidden_dim, batch_first=True))
            self.gru_document_a.append(nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True))

        self.attentions_f = AttentionTanH(config)
        self.attentionw_f = AttentionTanH(config)

        self.attentions_a = []
        self.attentionw_a = []
        for i in range(self.top_k):
            self.attentions_a.append(AttentionTanH(config))
            self.attentionw_a.append(AttentionTanH(config))
        self.attention_a = AttentionTanH(config)
        # task_name = config.get("data", "type_of_label").replace(" ", "").split(",")[0]
        # self.outfc = nn.Linear(150, get_num_classes(task_name))

        self.midfc1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        self.midfc2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        self.attfc_as = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.attfc_aw = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.attfc_ad = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.birnn = nn.RNN(self.hidden_dim, self.hidden_dim, batch_first=True)

        self.init_hidden(config, usegpu)

        self.gru_sentence_a = nn.ModuleList(self.gru_sentence_a)
        self.gru_document_a = nn.ModuleList(self.gru_document_a)
        self.attentions_a = nn.ModuleList(self.attentions_a)
        self.attentionw_a = nn.ModuleList(self.attentionw_a)

        self.outfc = []
        task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
        for x in task_name:
            self.outfc.append(nn.Linear(
                self.hidden_dim, get_num_classes(x)
            ))

        self.midfc = []
        for x in task_name:
            self.midfc.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        self.cell_list = [None]
        for x in task_name:
            self.cell_list.append(nn.LSTMCell(self.hidden_dim, self.hidden_dim))

        self.outfc = nn.ModuleList(self.outfc)
        self.midfc = nn.ModuleList(self.midfc)

        self.hidden_state_fc_list = []
        for a in range(0, len(task_name) + 1):
            arr = []
            for b in range(0, len(task_name) + 1):
                arr.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            arr = nn.ModuleList(arr)
            self.hidden_state_fc_list.append(arr)

        self.cell_state_fc_list = []
        for a in range(0, len(task_name) + 1):
            arr = []
            for b in range(0, len(task_name) + 1):
                arr.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            arr = nn.ModuleList(arr)
            self.cell_state_fc_list.append(arr)

        self.cell_list = nn.ModuleList(self.cell_list)
        self.hidden_state_fc_list = nn.ModuleList(self.hidden_state_fc_list)
        self.cell_state_fc_list = nn.ModuleList(self.cell_state_fc_list)

        # self.dropout = nn.Dropout(config.getfloat("train", "dropout"))

    def init_hidden(self, config, usegpu):
        if torch.cuda.is_available() and usegpu:
            self.sentence_hidden_f = torch.autograd.Variable(
                torch.zeros(1, config.getint("data", "batch_size") * config.getint("data", "sentence_num"),
                            self.hidden_dim).cuda())
            self.document_hidden_f = torch.autograd.Variable(
                torch.zeros(1, config.getint("data", "batch_size"), self.hidden_dim).cuda())
            self.sentence_hidden_a = []
            self.document_hidden_a = []
            for i in range(self.top_k):
                self.sentence_hidden_a.append(
                    torch.autograd.Variable(
                        torch.zeros(1, config.getint("data", "batch_size") * config.getint("data", "sentence_num"),
                                    self.hidden_dim).cuda()))
                self.document_hidden_a.append(torch.autograd.Variable(
                    torch.zeros(1, config.getint("data", "batch_size"), self.hidden_dim).cuda()))
            self.birnn_hidden = torch.autograd.Variable(
                torch.zeros(1, config.getint("data", "batch_size"), self.hidden_dim).cuda())

        else:
            self.sentence_hidden_f = torch.autograd.Variable(
                torch.zeros(1, config.getint("data", "batch_size") * config.getint("data", "sentence_num"),
                            self.hidden_dim))
            self.document_hidden_f = torch.autograd.Variable(
                torch.zeros(1, config.getint("data", "batch_size"), self.hidden_dim))
            self.sentence_hidden_a = []
            self.document_hidden_a = []
            for i in range(self.top_k):
                self.sentence_hidden_a.append(
                    torch.autograd.Variable(
                        torch.zeros(1, config.getint("data", "batch_size") * config.getint("data", "sentence_num"),
                                    self.hidden_dim)))
                self.document_hidden_a.append(
                    torch.autograd.Variable(torch.zeros(1, config.getint("data", "batch_size"), self.hidden_dim)))
            self.birnn_hidden = torch.autograd.Variable(
                torch.zeros(1, config.getint("data", "batch_size"), self.hidden_dim))

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

    def forward(self, x, doc_len, config, content):

        x = x.view(config.getint("data", "batch_size") * config.getint("data", "sentence_num"),
                   config.getint("data", "sentence_len"),
                   config.getint("data", "vec_size"))

        sentence_out, self.sentence_hidden_f = self.gru_sentence_f(x, self.sentence_hidden_f)
        # print(sentence_out.size())
        # sentence_out = self.attentionw_f(self.ufw, sentence_out)

        sentence_out = sentence_out.contiguous().view(
            config.getint("data", "batch_size"), config.getint("data", "sentence_num"),
            config.getint("data", "sentence_len"),
            config.getint("net", "hidden_size"))
        sentence_out = torch.max(sentence_out, dim=2)[0]
        sentence_out = sentence_out.view(
            config.getint("data", "batch_size"), config.getint("data", "sentence_num"),
            config.getint("net", "hidden_size"))

        sentence_out = sentence_out.view(config.getint("data", "batch_size"), config.getint("data", "sentence_num"),
                                         self.hidden_dim)

        doc_out, self.document_hidden_f = self.gru_document_f(sentence_out, self.document_hidden_f)

        # df = self.attentions_f(self.ufs, doc_out)
        doc_out = doc_out.contiguous().view(config.getint("data", "batch_size"), config.getint("data", "sentence_num"),
                                            config.getint("net", "hidden_size"))
        df = torch.max(doc_out, dim=1)[0]

        uas = self.attfc_as(df)
        uaw = self.attfc_aw(df)
        # print(uaw.size())
        uaw = torch.split(uaw, 1, dim=0)
        # print(uaw)
        tmp_uaw = []
        for i in range(config.getint("data", "batch_size")):
            for j in range(config.getint("data", "sentence_num")):
                tmp_uaw.append(uaw[i])
        # print(tmp_uaw)
        uaw = torch.cat(tmp_uaw, dim=0)
        uad = self.attfc_ad(df)

        out_art = []
        x_a = torch.unbind(x_a, dim=1)
        for i in range(self.top_k):
            x = x_a[i]
            x = x.contiguous().view(config.getint("data", "batch_size") * config.getint("data", "sentence_num"),
                                    config.getint("data", "sentence_len"), config.getint("data", "vec_size"))
            # print(x.size())
            sentence_out, self.sentence_hidden_a[i] = self.gru_sentence_a[i](x, self.sentence_hidden_a[i])

            # print(doc_len)
            sentence_out = self.attentionw_a[i](uaw, sentence_out)
            sentence_out = sentence_out.view(config.getint("data", "batch_size"), config.getint("data", "sentence_num"),
                                             self.hidden_dim)
            # print(sentence.size())
            doc_out, self.document_hidden_a[i] = self.gru_document_a[i](sentence_out, self.document_hidden_a[i])

            out_art.append(self.attentions_f(uas, doc_out))

        out_art = torch.cat(out_art).view(config.getint("data", "batch_size"), self.top_k, -1)

        out_art, self.birnn_hidden = self.birnn(out_art, self.birnn_hidden)
        # print(out_art.size())
        da = self.attention_a(uad, out_art)
        # print(df.size())
        # print(da.size())
        final_out = torch.cat((df, da), dim=1)
        # print(final_out.size())
        final_out = self.midfc2(F.relu(self.midfc1(final_out)))

        outputs = []
        task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
        graph = generate_graph(config)

        first = []
        for a in range(0, len(task_name) + 1):
            first.append(True)
        for a in range(1, len(task_name) + 1):
            h, c = self.cell_list[a](final_out, self.hidden_list[a])
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
                    # if config.getboolean("net", "attention"):
                    # h = self.attetion(h, attention_value)
            if config.getboolean("net", "more_fc"):
                outputs.append(
                    self.outfc[a - 1](F.relu(self.midfc[a - 1](h))).view(config.getint("data", "batch_size"), -1))
            else:
                outputs.append(self.outfc[a - 1](h).view(config.getint("data", "batch_size"), -1))

        return outputs
