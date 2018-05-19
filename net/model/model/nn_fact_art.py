import torch
import torch.nn as nn
import torch.nn.functional as F

from net.model.layer import AttentionTanH
from net.loader import get_num_classes
from net.model.layer import svm
from net.model.decoder import FCDecoder


class NNFactArt(nn.Module):
    def __init__(self, config, usegpu):
        super(NNFactArt, self).__init__()

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

        self.attentions_f = AttentionTanH(config)
        self.attentionw_f = AttentionTanH(config)

        self.attentions_a = []
        self.attentionw_a = []
        for i in range(self.top_k):
            self.attentions_a.append(AttentionTanH(config))
            self.attentionw_a.append(AttentionTanH(config))
        self.attention_a = AttentionTanH(config)
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
        self.svm = svm(config, usegpu)
        self.decoder = FCDecoder(config, usegpu)
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
        x_a = []
        for co in content:
            tmp = torch.stack(self.svm.top2law(config, co))
            x_a.append(tmp)
        x_a = torch.stack(x_a)
        # print(x_a)
        x_a = torch.unbind(x_a, dim=1)
        # print(x_a)
        for i in range(self.top_k):
            x = x_a[i]
            # print(x)
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
        outputs = self.decoder.forward(final_out, doc_len, config)
        # outputs = [self.outfc(F.relu(self.midfc2(F.relu(self.midfc1(final_out)))))]

        return outputs
