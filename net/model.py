import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import os

from utils import calc_accuracy, gen_result, get_num_classes, generate_graph
import pdb


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        pass
        # self.fc = nn.Linear(config.getint("net", "hidden_size"), config.getint("net", "hidden_size"))

    def forward(self, feature, hidden):
        feature = feature.view(feature.size(0), -1, 1)
        ratio = torch.bmm(hidden, feature)
        ratio = ratio.view(ratio.size(0), ratio.size(1))
        ratio = F.softmax(ratio, dim=1).view(ratio.size(0), -1, 1)
        result = torch.bmm(hidden.transpose(1, 2), ratio)
        result = result.view(result.size(0), -1)

        return result


class Attention_tanh(nn.Module):
    def __init__(self, config):
        super(Attention_tanh, self).__init__()
        # pass
        self.fc = nn.Linear(config.getint("net", "hidden_size"), config.getint("net", "hidden_size"), bias=False)

    def forward(self, feature, hidden):
        feature = feature.view(feature.size(0), -1, 1)
        hidden = torch.tanh(self.fc(hidden))
        # print(feature.size())
        # print(hidden.size())
        ratio = torch.bmm(hidden, feature)
        ratio = ratio.view(ratio.size(0), ratio.size(1))
        ratio = F.softmax(ratio, dim=1).view(ratio.size(0), -1, 1)
        result = torch.bmm(hidden.transpose(1, 2), ratio)
        result = result.view(result.size(0), -1)

        return result


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()

        self.convs = []

        for a in range(config.getint("net", "min_gram"), config.getint("net", "max_gram") + 1):
            self.convs.append(nn.Conv2d(1, config.getint("net", "filters"), (a, config.getint("data", "vec_size"))))

        features = (config.getint("net", "max_gram") - config.getint("net", "min_gram") + 1) * config.getint("net",
                                                                                                             "filters")
        self.outfc = []
        task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
        for x in task_name:
            self.outfc.append(nn.Linear(
                features, get_num_classes(x)
            ))

        self.midfc = []
        for x in task_name:
            self.midfc.append(nn.Linear(features, features))

        self.dropout = nn.Dropout(config.getfloat("train", "dropout"))
        self.convs = nn.ModuleList(self.convs)
        self.outfc = nn.ModuleList(self.outfc)
        self.midfc = nn.ModuleList(self.midfc)

    def init_hidden(self, config, usegpu):
        pass

    def forward(self, x, doc_len, config):
        x = x.view(config.getint("data", "batch_size"), 1, -1, config.getint("data", "vec_size"))
        fc_input = []
        for conv in self.convs:
            fc_input.append(torch.max(conv(x), dim=2, keepdim=True)[0])

        features = (config.getint("net", "max_gram") - config.getint("net", "min_gram") + 1) * config.getint("net",
                                                                                                             "filters")

        fc_input = torch.cat(fc_input, dim=1).view(-1, features)

        outputs = []
        now_cnt = 0
        for fc in self.outfc:
            if config.getboolean("net", "more_fc"):
                outputs.append(fc(F.relu(self.midfc[now_cnt](fc_input))))
            else:
                outputs.append(fc(fc_input))
            now_cnt += 1

        return outputs


class LSTM(nn.Module):
    def __init__(self, config, usegpu):
        super(LSTM, self).__init__()

        self.data_size = config.getint("data", "vec_size")
        self.hidden_dim = config.getint("net", "hidden_size")

        self.lstm = nn.LSTM(self.data_size, self.hidden_dim, batch_first=True)

        self.outfc = []
        task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
        for x in task_name:
            self.outfc.append(nn.Linear(
                self.hidden_dim, get_num_classes(x)
            ))

        self.midfc = []
        for x in task_name:
            self.midfc.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        self.dropout = nn.Dropout(config.getfloat("train", "dropout"))
        self.outfc = nn.ModuleList(self.outfc)
        self.init_hidden(config, usegpu)
        self.midfc = nn.ModuleList(self.midfc)

    def init_hidden(self, config, usegpu):
        if torch.cuda.is_available() and usegpu:
            self.hidden = (
                torch.autograd.Variable(torch.zeros(1, config.getint("data", "batch_size"), self.hidden_dim).cuda()),
                torch.autograd.Variable(torch.zeros(1, config.getint("data", "batch_size"), self.hidden_dim).cuda()))
        else:
            self.hidden = (
                torch.autograd.Variable(torch.zeros(1, config.getint("data", "batch_size"), self.hidden_dim)),
                torch.autograd.Variable(torch.zeros(1, config.getint("data", "batch_size"), self.hidden_dim)))

    def forward(self, x, doc_len, config):
        # x = x.view(config.getint("data", "batch_size"), 1, -1, config.getint("data", "vec_size"))

        x = x.view(config.getint("data", "batch_size"),
                   config.getint("data", "sentence_num") * config.getint("data", "sentence_len"),
                   config.getint("data", "vec_size"))

        lstm_out, self.hidden = self.lstm(x, self.hidden)
        # lstm_out = self.dropout(lstm_out)

        outv = []
        for a in range(0, len(doc_len)):
            outv.append(lstm_out[a][doc_len[a][0] - 1])
        lstm_out = torch.cat(outv)

        outputs = []
        now_cnt = 0
        for fc in self.outfc:
            if config.getboolean("net", "more_fc"):
                outputs.append(fc(F.relu(self.midfc[now_cnt](lstm_out))))
            else:
                outputs.append(fc(lstm_out))
            now_cnt += 1

        return outputs


class MULTI_LSTM(nn.Module):
    def __init__(self, config, usegpu):
        super(MULTI_LSTM, self).__init__()

        self.data_size = config.getint("data", "vec_size")
        self.hidden_dim = config.getint("net", "hidden_size")

        self.lstm_sentence = nn.LSTM(self.data_size, self.hidden_dim, batch_first=True)
        self.lstm_document = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)

        self.outfc = []
        task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
        for x in task_name:
            self.outfc.append(nn.Linear(
                self.hidden_dim, get_num_classes(x)
            ))

        self.midfc = []
        for x in task_name:
            self.midfc.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        self.dropout = nn.Dropout(config.getfloat("train", "dropout"))
        self.outfc = nn.ModuleList(self.outfc)
        self.init_hidden(config, usegpu)
        self.midfc = nn.ModuleList(self.midfc)

    def init_hidden(self, config, usegpu):
        if torch.cuda.is_available() and usegpu:
            self.sentence_hidden = (
                torch.autograd.Variable(
                    torch.zeros(1, config.getint("data", "batch_size") * config.getint("data", "sentence_num"),
                                self.hidden_dim).cuda()),
                torch.autograd.Variable(
                    torch.zeros(1, config.getint("data", "batch_size") * config.getint("data", "sentence_num"),
                                self.hidden_dim).cuda()))
            self.document_hidden = (
                torch.autograd.Variable(torch.zeros(1, config.getint("data", "batch_size"), self.hidden_dim).cuda()),
                torch.autograd.Variable(torch.zeros(1, config.getint("data", "batch_size"), self.hidden_dim).cuda()))
        else:
            self.sentence_hidden = (
                torch.autograd.Variable(
                    torch.zeros(1, config.getint("data", "batch_size") * config.getint("data", "sentence_num"),
                                self.hidden_dim)),
                torch.autograd.Variable(
                    torch.zeros(1, config.getint("data", "batch_size") * config.getint("data", "sentence_num"),
                                self.hidden_dim)))
            self.document_hidden = (
                torch.autograd.Variable(torch.zeros(1, config.getint("data", "batch_size"), self.hidden_dim)),
                torch.autograd.Variable(torch.zeros(1, config.getint("data", "batch_size"), self.hidden_dim)))

    def forward(self, x, doc_len, config):

        x = x.view(config.getint("data", "batch_size") * config.getint("data", "sentence_num"),
                   config.getint("data", "sentence_len"),
                   config.getint("data", "vec_size"))

        sentence_out, self.sentence_hidden = self.lstm_sentence(x, self.sentence_hidden)
        temp_out = []
        # print(doc_len)
        if config.get("net", "method") == "LAST":
            for a in range(0, len(sentence_out)):
                idx = a // config.getint("data", "sentence_num")
                idy = a % config.getint("data", "sentence_num")
                # print(idx,idy)
                temp_out.append(sentence_out[a][doc_len[idx][idy + 2] - 1])
            sentence_out = torch.stack(temp_out)
        elif config.get("net", "method") == "MAX":
            sentence_out = sentence_out.contiguous().view(
                config.getint("data", "batch_size"), config.getint("data", "sentence_num"),
                config.getint("data", "sentence_len"),
                config.getint("net", "hidden_size"))
            sentence_out = torch.max(sentence_out, dim=2)[0]
            sentence_out = sentence_out.view(
                config.getint("data", "batch_size"), config.getint("data", "sentence_num"),
                config.getint("net", "hidden_size"))
        else:
            gg
        sentence_out = sentence_out.view(config.getint("data", "batch_size"), config.getint("data", "sentence_num"),
                                         self.hidden_dim)

        lstm_out, self.document_hidden = self.lstm_document(sentence_out, self.document_hidden)

        if config.get("net", "method") == "LAST":
            outv = []
            for a in range(0, len(doc_len)):
                outv.append(lstm_out[a][doc_len[a][1] - 1])
            lstm_out = torch.cat(outv)
        elif config.get("net", "method") == "MAX":
            lstm_out = torch.max(lstm_out, dim=1)[0]
        else:
            gg

        outputs = []
        now_cnt = 0
        for fc in self.outfc:
            if config.getboolean("net", "more_fc"):
                outputs.append(fc(F.relu(self.midfc[now_cnt](lstm_out))))
            else:
                outputs.append(fc(lstm_out))
            now_cnt += 1

        return outputs


class NN_fact_art(nn.Module):
    def __init__(self, config, usegpu):
        super(NN_fact_art, self).__init__()

        self.data_size = config.getint("data", "vec_size")
        self.hidden_dim = config.getint("net", "hidden_size")
        self.top_k = config.getint("data", "top_k")

        if(usegpu):
            self.ufs = torch.autograd.Variable(torch.randn(config.getint("data", "batch_size"), self.hidden_dim)).cuda()
            self.ufw = torch.autograd.Variable(torch.randn(config.getint("data", "batch_size")  * config.getint("data", "sentence_num"), self.hidden_dim)).cuda()
        else:
            self.ufs = torch.autograd.Variable(torch.randn(config.getint("data", "batch_size"), self.hidden_dim))
            self.ufw = torch.autograd.Variable(torch.randn(config.getint("data", "batch_size")  * config.getint("data", "sentence_num"), self.hidden_dim))

        self.gru_sentence_f = nn.GRU(self.data_size, self.hidden_dim, batch_first=True)
        self.gru_document_f = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)

        self.gru_sentence_a = []
        self.gru_document_a = []
        for i in range(self.top_k):
            self.gru_sentence_a.append(nn.GRU(self.data_size, self.hidden_dim, batch_first=True))
            self.gru_document_a.append(nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True))

        self.attentions_f = Attention_tanh(config)
        self.attentionw_f = Attention_tanh(config)

        self.attentions_a = []
        self.attentionw_a = []
        for i in range(self.top_k):
            self.attentions_a.append(Attention_tanh(config))
            self.attentionw_a.append(Attention_tanh(config))
        self.attention_a = Attention_tanh(config)
        task_name = config.get("data", "type_of_label").replace(" ", "").split(",")[0]
        self.outfc = nn.Linear(150, get_num_classes(task_name))

        self.midfc1 = nn.Linear(self.hidden_dim * 2, 200)
        self.midfc2 = nn.Linear(200, 150)

        self.attfc_as = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.attfc_aw = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.attfc_ad = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.birnn = nn.RNN(self.hidden_dim, self.hidden_dim, batch_first=True)

        self.init_hidden(config, usegpu)


        self.gru_sentence_a = nn.ModuleList(self.gru_sentence_a)
        self.gru_document_a = nn.ModuleList(self.gru_document_a)
        self.attentions_a = nn.ModuleList(self.attentions_a)
        self.attentionw_a = nn.ModuleList(self.attentionw_a)
        # self.dropout = nn.Dropout(config.getfloat("train", "dropout"))
        # self.outfc = nn.ModuleList(self.outfc)

        # self.midfc = nn.ModuleList(self.midfc)

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

    def forward(self, x, x_a, doc_len, config):

        x = x.view(config.getint("data", "batch_size") * config.getint("data", "sentence_num"),
                   config.getint("data", "sentence_len"),
                   config.getint("data", "vec_size"))

        sentence_out, self.sentence_hidden_f = self.gru_sentence_f(x, self.sentence_hidden_f)
        # print(sentence_out.size())
        sentence_out = self.attentionw_f(self.ufw, sentence_out)
        sentence_out = sentence_out.view(config.getint("data", "batch_size"), config.getint("data", "sentence_num"),
                                         self.hidden_dim)

        doc_out, self.document_hidden_f = self.gru_document_f(sentence_out, self.document_hidden_f)

        df = self.attentions_f(self.ufs, doc_out)

        uas = self.attfc_as(df)
        uaw = self.attfc_aw(df)
        uaw = torch.cat([uaw for i in range(config.getint("data", "sentence_num"))])
        uad = self.attfc_ad(df)

        out_art = []
        x_a = torch.unbind(x_a, dim=1)
        for i in range(self.top_k):
            x = x_a[i]
            x = x.contiguous().view(config.getint("data", "batch_size") * config.getint("data", "sentence_num"), config.getint("data", "sentence_len"), config.getint("data", "vec_size"))
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
        outputs = [self.outfc(F.relu(self.midfc2(F.relu(self.midfc1(final_out)))))]

        return outputs


class CNN_FINAL(nn.Module):
    def __init__(self, config):
        super(CNN_FINAL, self).__init__()

        self.convs = []
        self.hidden_dim = config.getint("net", "hidden_size")

        for a in range(config.getint("net", "min_gram"), config.getint("net", "max_gram") + 1):
            self.convs.append(nn.Conv2d(1, config.getint("net", "filters"), (a, config.getint("data", "vec_size"))))

        features = (config.getint("net", "max_gram") - config.getint("net", "min_gram") + 1) * config.getint("net",
                                                                                                             "filters")
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

        self.dropout = nn.Dropout(config.getfloat("train", "dropout"))
        self.convs = nn.ModuleList(self.convs)
        self.outfc = nn.ModuleList(self.outfc)
        self.midfc = nn.ModuleList(self.midfc)
        self.cell_list = nn.ModuleList(self.cell_list)
        self.hidden_state_fc_list = nn.ModuleList(self.hidden_state_fc_list)
        self.cell_state_fc_list = nn.ModuleList(self.cell_state_fc_list)

    def init_hidden(self, config, usegpu):
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

    def forward(self, x, doc_len, config):
        x = x.view(config.getint("data", "batch_size"), 1, -1, config.getint("data", "vec_size"))
        conv_out = []

        for conv in self.convs:
            y = conv(x).view(config.getint("data", "batch_size"), config.getint("net", "filteres"), -1)
            y = F.pad(y,
                      (0, config.getint("data", "sentence_num") * config.getint("data", "sentence_len") - len(y[0][0])))
            conv_out.append(y)

        conv_out = torch.cat(conv_out, dim=1)
        fc_input = torch.max(conv_out, dim=1)[0]
        print(fc_input)
        gg

        features = (config.getint("net", "max_gram") - config.getint("net", "min_gram") + 1) * config.getint("net",
                                                                                                             "filters")

        fc_input = fc_input.view(-1, features)

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
            if config.getboolean("net", "more_fc"):
                outputs.append(
                    self.outfc[a - 1](F.relu(self.midfc[a - 1](h))).view(config.getint("data", "batch_size"), -1))
            else:
                outputs.append(self.outfc[a - 1](h).view(config.getint("data", "batch_size"), -1))

        return outputs


class MULTI_LSTM_FINAL(nn.Module):
    def __init__(self, config, usegpu):
        super(MULTI_LSTM_FINAL, self).__init__()

        self.data_size = config.getint("data", "vec_size")
        self.hidden_dim = config.getint("net", "hidden_size")

        self.lstm_sentence = nn.LSTM(self.data_size, self.hidden_dim, batch_first=True)
        self.lstm_document = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)

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
            self.cell_list.append(nn.LSTMCell(config.getint("net", "hidden_size"), config.getint("net", "hidden_size")))

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

        self.attetion = Attention(config)

        self.dropout = nn.Dropout(config.getfloat("train", "dropout"))
        self.outfc = nn.ModuleList(self.outfc)
        self.init_hidden(config, usegpu)
        self.midfc = nn.ModuleList(self.midfc)
        self.cell_list = nn.ModuleList(self.cell_list)
        self.hidden_state_fc_list = nn.ModuleList(self.hidden_state_fc_list)
        self.cell_state_fc_list = nn.ModuleList(self.cell_state_fc_list)

    def init_hidden(self, config, usegpu):
        if torch.cuda.is_available() and usegpu:
            self.sentence_hidden = (
                torch.autograd.Variable(
                    torch.zeros(1, config.getint("data", "batch_size") * config.getint("data", "sentence_num"),
                                self.hidden_dim).cuda()),
                torch.autograd.Variable(
                    torch.zeros(1, config.getint("data", "batch_size") * config.getint("data", "sentence_num"),
                                self.hidden_dim).cuda()))
            self.document_hidden = (
                torch.autograd.Variable(torch.zeros(1, config.getint("data", "batch_size"), self.hidden_dim).cuda()),
                torch.autograd.Variable(torch.zeros(1, config.getint("data", "batch_size"), self.hidden_dim).cuda()))
        else:
            self.sentence_hidden = (
                torch.autograd.Variable(
                    torch.zeros(1, config.getint("data", "batch_size") * config.getint("data", "sentence_num"),
                                self.hidden_dim)),
                torch.autograd.Variable(
                    torch.zeros(1, config.getint("data", "batch_size") * config.getint("data", "sentence_num"),
                                self.hidden_dim)))
            self.document_hidden = (
                torch.autograd.Variable(torch.zeros(1, config.getint("data", "batch_size"), self.hidden_dim)),
                torch.autograd.Variable(torch.zeros(1, config.getint("data", "batch_size"), self.hidden_dim)))

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

    def forward(self, x, doc_len, config):

        x = x.view(config.getint("data", "batch_size") * config.getint("data", "sentence_num"),
                   config.getint("data", "sentence_len"),
                   config.getint("data", "vec_size"))

        sentence_out, self.sentence_hidden = self.lstm_sentence(x, self.sentence_hidden)
        # print(doc_len)
        if config.get("net", "method") == "LAST":
            temp_out = []
            for a in range(0, len(sentence_out)):
                idx = a // config.getint("data", "sentence_num")
                idy = a % config.getint("data", "sentence_num")
                # print(idx,idy)
                temp_out.append(sentence_out[a][doc_len[idx][idy + 2] - 1])
            sentence_out = torch.stack(temp_out)
        elif config.get("net", "method") == "MAX":
            sentence_out = sentence_out.contiguous().view(
                config.getint("data", "batch_size"), config.getint("data", "sentence_num"),
                config.getint("data", "sentence_len"),
                config.getint("net", "hidden_size"))
            sentence_out = torch.max(sentence_out, dim=2)[0]
            sentence_out = sentence_out.view(
                config.getint("data", "batch_size"), config.getint("data", "sentence_num"),
                config.getint("net", "hidden_size"))
            """lx = len(sentence_out)
            sentence_out = sentence_out.contiguous().view(
                config.getint("data", "batch_size"), config.getint("data", "sentence_num"),
                config.getint("data", "sentence_len"),
                config.getint("net", "hidden_size"))
            temp_out = []
            for a in range(0, lx):
                idx = a // config.getint("data", "sentence_num")
                idy = a % config.getint("data", "sentence_num")
                ly = doc_len[idx][idy+2].data[0]
                temp_out.append(torch.max(sentence_out[idx,idy,0:ly+1,:],dim=0)[0])
            sentence_out = torch.stack(temp_out)"""
        else:
            gg
        sentence_out = sentence_out.view(config.getint("data", "batch_size"), config.getint("data", "sentence_num"),
                                         self.hidden_dim)

        lstm_out, self.document_hidden = self.lstm_document(sentence_out, self.document_hidden)
        attention_value = lstm_out

        if config.get("net", "method") == "LAST":
            outv = []
            for a in range(0, len(doc_len)):
                outv.append(lstm_out[a][doc_len[a][1] - 1])
            lstm_out = torch.cat(outv)
        elif config.get("net", "method") == "MAX":
            outv = []
            for a in range(0, len(doc_len)):
                outv.append(torch.max(lstm_out[a, 0:doc_len[a][1].data[0]], dim=0)[0])
            lstm_out = torch.stack(outv)
            # lstm_out = torch.max(lstm_out, dim=1)[0]
        else:
            gg

        outputs = []
        task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
        graph = generate_graph(config)

        first = []
        for a in range(0, len(task_name) + 1):
            first.append(True)
        for a in range(1, len(task_name) + 1):
            h, c = self.cell_list[a](lstm_out, self.hidden_list[a])
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
            if config.getboolean("net", "attention"):
                h = self.attetion(h, attention_value)
            if config.getboolean("net", "more_fc"):
                outputs.append(
                    self.outfc[a - 1](F.relu(self.midfc[a - 1](h))).view(config.getint("data", "batch_size"), -1))
            else:
                outputs.append(self.outfc[a - 1](h).view(config.getint("data", "batch_size"), -1))

        """previous version
        for a in range(1, len(task_name) + 1):
            h, c = self.cell_list[a](lstm_out, self.hidden_list[a - 1])
            h = h + self.combine_fc_list[a](lstm_out)
            self.hidden_list[a] = h, c
            if config.getboolean("net", "more_fc"):
                outputs.append(
                    self.outfc[a - 1](F.relu(self.midfc[a - 1](h))).view(config.getint("data", "batch_size", -1)))
            else:
                outputs.append(self.outfc[a - 1](h).view(config.getint("data", "batch_size"), -1))
        """

        return outputs


def test(net, test_dataset, usegpu, config, epoch):
    net.eval()
    running_acc = []
    task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
    batch_size = config.getint("data", "batch_size")
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

    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=1)
    for idx, data in enumerate(test_data_loader):
        inputs, doc_len, labels = data

        net.init_hidden(config, usegpu)

        if torch.cuda.is_available() and usegpu:
            inputs, doc_len, labels = Variable(inputs.cuda()), Variable(doc_len.cuda()), Variable(labels.cuda())
        else:
            inputs, doc_len, labels = Variable(inputs), Variable(doc_len), Variable(labels)

        outputs = net.forward(inputs, doc_len, config)
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


def train(net, train_dataset, test_dataset, usegpu, config):
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
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-8)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momemtum)
    else:
        gg

    total_loss = []

    print("Training begin")
    net.train()
    first = True

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
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True,
                                       num_workers=1)
        for idx, data in enumerate(train_data_loader):
            cnt += 1
            inputs, doc_len, labels = data
            # print("inputs",inputs)
            # print("doc_len",doc_len)
            # print("labels",labels)
            if torch.cuda.is_available() and usegpu:
                inputs, doc_len, labels = Variable(inputs.cuda()), Variable(doc_len.cuda()), Variable(labels.cuda())
            else:
                inputs, doc_len, labels = Variable(inputs), Variable(doc_len), Variable(labels)

            net.init_hidden(config, usegpu)
            optimizer.zero_grad()

            outputs = net.forward(inputs, doc_len, config)
            losses = []
            for a in range(0, len(task_name)):
                losses.append(criterion(outputs[a], labels.transpose(0, 1)[a]))
                running_acc[a] = calc_accuracy(outputs[a], labels.transpose(0, 1)[a], running_acc[a])
            loss = torch.sum(torch.stack(losses))

            loss.backward()
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

        test(net, test_dataset, usegpu, config, epoch_num + 1)
        if not (os.path.exists(model_path)):
            os.makedirs(model_path)
        torch.save(net.state_dict(), os.path.join(model_path, "model-%d.pkl" % (epoch_num + 1)))

    print("Training done")

    test(net, test_dataset, usegpu, config, 0)
    torch.save(net.state_dict(), os.path.join(model_path, "model-0.pkl"))

    return net


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

        inputs, doc_len, labels = data

        net.init_hidden(config, usegpu)

        if torch.cuda.is_available() and usegpu:
            inputs, doc_len, labels = Variable(inputs.cuda()), Variable(doc_len.cuda()), Variable(labels.cuda())
        else:
            inputs, doc_len, labels = Variable(inputs), Variable(doc_len), Variable(labels)

        outputs = net.forward(inputs, doc_len, config)
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

            inputs, doc_len, labels = data
            if torch.cuda.is_available() and usegpu:
                inputs, doc_len, labels = Variable(inputs.cuda()), Variable(doc_len.cuda()), Variable(labels.cuda())
            else:
                inputs, doc_len, labels = Variable(inputs), Variable(doc_len), Variable(labels)

            net.init_hidden(config, usegpu)
            optimizer.zero_grad()

            outputs = net.forward(inputs, doc_len, config)
            loss = 0
            for a in range(0, len(task_name)):
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
