import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, config, usegpu):
        super(LSTMEncoder, self).__init__()

        self.data_size = config.getint("data", "vec_size")
        self.hidden_dim = config.getint("net", "hidden_size")

        self.lstm_sentence = nn.LSTM(self.data_size, self.hidden_dim, batch_first=True,
                                     num_layers=config.getint("net", "num_layers"))
        self.lstm_document = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True,
                                     num_layers=config.getint("net", "num_layers"))
        self.feature_len = self.hidden_dim

    def init_hidden(self, config, usegpu):
        if torch.cuda.is_available() and usegpu:
            self.sentence_hidden = (
                torch.autograd.Variable(
                    torch.zeros(config.getint("net", "num_layers"),
                                config.getint("data", "batch_size") * config.getint("data", "sentence_num"),
                                self.hidden_dim).cuda()),
                torch.autograd.Variable(
                    torch.zeros(config.getint("net", "num_layers"),
                                config.getint("data", "batch_size") * config.getint("data", "sentence_num"),
                                self.hidden_dim).cuda()))
            self.document_hidden = (
                torch.autograd.Variable(
                    torch.zeros(config.getint("net", "num_layers"), config.getint("data", "batch_size"),
                                self.hidden_dim).cuda()),
                torch.autograd.Variable(
                    torch.zeros(config.getint("net", "num_layers"), config.getint("data", "batch_size"),
                                self.hidden_dim).cuda()))
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
