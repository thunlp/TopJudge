import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import os
import configparser

from utils import calc_accuracy, gen_result, get_num_classes, generate_graph, generate_article_list
import pdb


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        pass

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


class CNN_ENCODER(nn.Module):
    def __init__(self, config, usegpu):
        super(CNN_ENCODER, self).__init__()

        self.convs = []
        for a in range(config.getint("net", "min_gram"), config.getint("net", "max_gram") + 1):
            self.convs.append(nn.Conv2d(1, config.getint("net", "filters"), (a, config.getint("data", "vec_size"))))

        self.convs = nn.ModuleList(self.convs)
        self.feature_len = (-config.getint("net", "min_gram") + config.getint("net", "max_gram") + 1) * config.getint(
            "net", "filters")

    def forward(self, x, doc_len, config):
        x = x.view(config.getint("data", "batch_size"), 1, -1, config.getint("data", "vec_size"))
        conv_out = []

        for conv in self.convs:
            # print("gg",type(x))
            y = conv(x).view(config.getint("data", "batch_size"), config.getint("net", "filters"), -1)
            y = F.pad(y,
                      (0, config.getint("data", "sentence_num") * config.getint("data", "sentence_len") - len(y[0][0])))
            conv_out.append(y)

        conv_out = torch.cat(conv_out, dim=1)
        conv_out = conv_out.view(config.getint("data", "batch_size"), -1,
                                 config.getint("data", "sentence_num") * config.getint("data", "sentence_len"))

        self.attention = conv_out.transpose(1, 2)
        # print(conv_out)
        fc_input = torch.max(conv_out, dim=2)[0]
        # print(fc_input)

        features = (config.getint("net", "max_gram") - config.getint("net", "min_gram") + 1) * config.getint("net",
                                                                                                             "filters")

        fc_input = fc_input.view(-1, features)
        # print(fc_input)

        return fc_input


class LSTM_SINGLE_ENCODER(nn.Module):
    def __init__(self, config, usegpu):
        super(LSTM_SINGLE_ENCODER, self).__init__()

        self.data_size = config.getint("data", "vec_size")
        self.hidden_dim = config.getint("net", "hidden_size")

        self.lstm = nn.LSTM(self.data_size, self.hidden_dim, batch_first=True)

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
        x = x.view(config.getint("data", "batch_size"),
                   config.getint("data", "sentence_num") * config.getint("data", "sentence_len"),
                   config.getint("data", "vec_size"))

        lstm_out, self.hidden = self.lstm(x, self.hidden)

        self.attention = lstm_out
        self.attention = torch.transpose(1, 2)
        print(self.attention)
        if config.get("net", "method") == "LAST":
            outv = []
            for a in range(0, len(doc_len)):
                outv.append(lstm_out[a][doc_len[a][0] - 1])
            lstm_out = torch.cat(outv)
        elif config.get("net", "method") == "MAX":
            lstm_out = torch.max(lstm_out, dim=1)[0]
            print(lstm_out)
            gg
        else:
            gg

        return lstm_out


class LSTM_ENCODER(nn.Module):
    def __init__(self, config, usegpu):
        super(LSTM_ENCODER, self).__init__()

        self.data_size = config.getint("data", "vec_size")
        self.hidden_dim = config.getint("net", "hidden_size")

        self.lstm_sentence = nn.LSTM(self.data_size, self.hidden_dim, batch_first=True)
        self.lstm_document = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.feature_len = self.hidden_dim

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
        if config.get("net", "method") == "LAST":
            for a in range(0, len(sentence_out)):
                idx = a // config.getint("data", "sentence_num")
                idy = a % config.getint("data", "sentence_num")
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

        self.attention = lstm_out

        if config.get("net", "method") == "LAST":
            outv = []
            for a in range(0, len(doc_len)):
                outv.append(lstm_out[a][doc_len[a][1] - 1])
            lstm_out = torch.cat(outv)
        elif config.get("net", "method") == "MAX":
            lstm_out = torch.max(lstm_out, dim=1)[0]
        else:
            gg

        return lstm_out


class ARTICLE_ENCODER(nn.Module):
    def __init__(self, config, usegpu):
        super(ARTICLE_ENCODER, self).__init__()

        self.article_encoder = CNN_ENCODER(config, usegpu)
        self.falv_list = generate_article_list(config, usegpu)

    def init_hidden(self, config, usegpu):
        pass

    def forward(self, x, doc_len, config):
        idx = torch.max(x, dim=1)[1]
        x = []
        doc_len = []
        for a in range(0, len(idx)):
            data = self.falv_list[idx[a].data[0]]
            x.append(data[0])
            doc_len.append(data[1])
        x = torch.stack(x)
        doc_len = torch.stack(doc_len)
        x = self.article_encoder(x, doc_len, config)
        return x


class FC_DECODER(nn.Module):
    def __init__(self, config, usegpu):
        super(FC_DECODER, self).__init__()
        try:
            features = (config.getint("net", "max_gram") - config.getint("net", "min_gram") + 1) * config.getint("net",
                                                                                                                 "filters")
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


class LSTM_DECODER(nn.Module):
    def __init__(self, config, usegpu):
        super(LSTM_DECODER, self).__init__()
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

        return outputs


class LSTM_DECODER_ARTICLE(nn.Module):
    def __init__(self, config, usegpu):
        super(LSTM_DECODER_ARTICLE, self).__init__()
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

        self.article_encoder = ARTICLE_ENCODER(config, usegpu)
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


class ARTICLE(nn.Module):
    def __init__(self, config, usegpu):
        super(ARTICLE, self).__init__()

        self.encoder = CNN_ENCODER(config, usegpu)
        self.decoder = LSTM_DECODER_ARTICLE(config, usegpu)

    def init_hidden(self, config, usegpu):
        self.decoder.init_hidden(config, usegpu)

    def forward(self, x, doc_len, config):
        x = self.encoder(x, doc_len, config)
        x = self.decoder(x, doc_len, config, self.encoder.attention)

        return x


class CNN(nn.Module):
    def __init__(self, config, usegpu):
        super(CNN, self).__init__()

        self.encoder = CNN_ENCODER(config, usegpu)
        self.decoder = FC_DECODER(config, usegpu)

    def init_hidden(self, config, usegpu):
        pass

    def forward(self, x, doc_len, config):
        x = self.encoder(x, doc_len, config)
        x = self.decoder(x, doc_len, config)

        return x


class LSTM(nn.Module):
    def __init__(self, config, usegpu):
        super(LSTM, self).__init__()

        self.encoder = LSTM_SINGLE_ENCODER(config, usegpu)
        self.decoder = FC_DECODER(config, usegpu)

    def init_hidden(self, config, usegpu):
        self.encoder.init_hidden(config, usegpu)

    def forward(self, x, doc_len, config):
        x = self.encoder(x, doc_len, config)
        x = self.decoder(x, doc_len, config)

        return x


class MULTI_LSTM(nn.Module):
    def __init__(self, config, usegpu):
        super(MULTI_LSTM, self).__init__()

        self.encoder = LSTM_ENCODER(config, usegpu)
        self.decoder = FC_DECODER(config, usegpu)

    def init_hidden(self, config, usegpu):
        self.encoder.init_hidden(config, usegpu)

    def forward(self, x, doc_len, config):
        x = self.encoder(x, doc_len, config)
        x = self.decoder(x, doc_len, config)

        return x


class CNN_FINAL(nn.Module):
    def __init__(self, config, usegpu):
        super(CNN_FINAL, self).__init__()

        self.encoder = CNN_ENCODER(config, usegpu)
        self.decoder = LSTM_DECODER(config, usegpu)
        self.trans_linear = nn.Linear(self.encoder.feature_len, self.decoder.feature_len)

    def init_hidden(self, config, usegpu):
        self.decoder.init_hidden(config, usegpu)

    def forward(self, x, doc_len, config):
        x = self.encoder(x, doc_len, config)
        if self.encoder.feature_len != self.decoder.feature_len:
            # print(self.encoder.feature_len,self.decoder.feature_len)
            x = self.trans_linear(x)
        x = self.decoder(x, doc_len, config, self.encoder.attention)

        return x


class MULTI_LSTM_FINAL(nn.Module):
    def __init__(self, config, usegpu):
        super(MULTI_LSTM_FINAL, self).__init__()

        self.encoder = LSTM_ENCODER(config, usegpu)
        self.decoder = LSTM_DECODER(config, usegpu)
        self.trans_linear = nn.Linear(self.encoder.feature_len, self.decoder.feature_len)

    def init_hidden(self, config, usegpu):
        self.encoder.init_hidden(config, usegpu)
        self.decoder.init_hidden(config, usegpu)

    def forward(self, x, doc_len, config):
        x = self.encoder(x, doc_len, config)
        if self.encoder.feature_len != self.decoder.feature_len:
            x = self.trans_linear(x)
        x = self.decoder(x, doc_len, config, self.encoder.attention)

        return x


class NN_fact_art(nn.Module):
    def __init__(self, config, usegpu):
        super(NN_fact_art, self).__init__()

        self.data_size = config.getint("data", "vec_size")
        self.hidden_dim = config.getint("net", "hidden_size")
        self.top_k = config.getint("data", "top_k")
        self.ufs = torch.ones(1, self.hidden_dim)
        # self.ufs = torch.randn(1, self.hidden_dim)
        self.ufs = torch.cat([self.ufs for i in range(config.getint("data", "batch_size"))], dim=0)
        self.ufw = torch.ones(1, self.hidden_dim)
        # self.ufw = torch.randn(1, self.hidden_dim)
        self.ufw = torch.cat(
            [self.ufw for i in range(config.getint("data", "batch_size") * config.getint("data", "sentence_num"))],
            dim=0)
        if (usegpu):
            self.ufs = torch.autograd.Variable(self.ufs).cuda()
            self.ufw = torch.autograd.Variable(self.ufw).cuda()
        else:
            self.ufs = torch.autograd.Variable(self.ufs)
            self.ufw = torch.autograd.Variable(self.ufw)

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
        outputs = [self.outfc(F.relu(self.midfc2(F.relu(self.midfc1(final_out)))))]

        return outputs


class NN_fact_art_final(nn.Module):
    def __init__(self, config, usegpu):
        super(NN_fact_art_final, self).__init__()

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

        self.attentions_f = Attention_tanh(config)
        self.attentionw_f = Attention_tanh(config)

        self.attentions_a = []
        self.attentionw_a = []
        for i in range(self.top_k):
            self.attentions_a.append(Attention_tanh(config))
            self.attentionw_a.append(Attention_tanh(config))
        self.attention_a = Attention_tanh(config)
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

    def forward(self, x, x_a, doc_len, config):

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
        torch.save(net, os.path.join(model_path, "model-%d.pkl" % (epoch_num + 1)))

    print("Training done")

    test(net, test_dataset, usegpu, config, 0)
    torch.save(net, os.path.join(model_path, "model-0.pkl"))

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
        torch.save(net, os.path.join(model_path, "model-%d.pkl" % (epoch_num + 1)))

    print("Training done")

    test_file(net, test_dataset, usegpu, config, 0)
    torch.save(net.state_dict(), os.path.join(model_path, "model.pkl"))
