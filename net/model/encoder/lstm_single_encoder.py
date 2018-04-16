import torch
import torch.nn as nn


class LSTMSingleEncoder(nn.Module):
    def __init__(self, config, usegpu):
        super(LSTMSingleEncoder, self).__init__()

        self.data_size = config.getint("data", "vec_size")
        self.hidden_dim = config.getint("net", "hidden_size")

        self.lstm = nn.LSTM(self.data_size, self.hidden_dim, batch_first=True,
                            num_layers=config.getint("net", "num_layers"))

    def init_hidden(self, config, usegpu):
        if torch.cuda.is_available() and usegpu:
            self.hidden = (
                torch.autograd.Variable(
                    torch.zeros(config.getint("net", "num_layers"), config.getint("data", "batch_size"),
                                self.hidden_dim).cuda()),
                torch.autograd.Variable(
                    torch.zeros(config.getint("net", "num_layers"), config.getint("data", "batch_size"),
                                self.hidden_dim).cuda()))
        else:
            self.hidden = (
                torch.autograd.Variable(
                    torch.zeros(config.getint("net", "num_layers"), config.getint("data", "batch_size"),
                                self.hidden_dim)),
                torch.autograd.Variable(
                    torch.zeros(config.getint("net", "num_layers"), config.getint("data", "batch_size"),
                                self.hidden_dim)))

    def forward(self, x, doc_len, config):
        x = x.view(config.getint("data", "batch_size"),
                   config.getint("data", "sentence_num") * config.getint("data", "sentence_len"),
                   config.getint("data", "vec_size"))

        lstm_out, self.hidden = self.lstm(x, self.hidden)

        self.attention = lstm_out
        if config.get("net", "method") == "LAST":
            outv = []
            for a in range(0, len(doc_len)):
                outv.append(lstm_out[a][doc_len[a][0] - 1])
            lstm_out = torch.cat(outv)
        elif config.get("net", "method") == "MAX":
            lstm_out = torch.max(lstm_out, dim=1)[0]
        else:
            gg

        return lstm_out
