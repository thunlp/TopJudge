import torch
import torch.nn as nn

from net.model.encoder import LSTMSingleEncoder
from net.model.decoder import FCDecoder


class LSTM(nn.Module):
    def __init__(self, config, usegpu):
        super(LSTM, self).__init__()

        self.encoder = LSTMSingleEncoder(config, usegpu)
        self.decoder = FCDecoder(config, usegpu)
        self.dropout = nn.Dropout(config.getfloat("train", "dropout"))

    def init_hidden(self, config, usegpu):
        self.encoder.init_hidden(config, usegpu)

    def forward(self, x, doc_len, config):
        x = self.encoder(x, doc_len, config)
        x = self.dropout(x)
        x = self.decoder(x, doc_len, config)

        return x
