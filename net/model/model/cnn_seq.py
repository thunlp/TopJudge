import torch
import torch.nn as nn

from net.model.encoder import CNNEncoder
from net.model.decoder import LSTMDecoder


class CNNSeq(nn.Module):
    def __init__(self, config, usegpu):
        super(CNNSeq, self).__init__()

        self.encoder = CNNEncoder(config, usegpu)
        self.decoder = LSTMDecoder(config, usegpu)
        self.trans_linear = nn.Linear(self.encoder.feature_len, self.decoder.feature_len)
        self.dropout = nn.Dropout(config.getfloat("train", "dropout"))

    def init_hidden(self, config, usegpu):
        self.decoder.init_hidden(config, usegpu)

    def forward(self, x, doc_len, config):
        x = self.encoder(x, doc_len, config)
        if self.encoder.feature_len != self.decoder.feature_len:
            # print(self.encoder.feature_len,self.decoder.feature_len)
            x = self.trans_linear(x)
        x = self.dropout(x)
        x = self.decoder(x, doc_len, config, self.encoder.attention)

        return x
