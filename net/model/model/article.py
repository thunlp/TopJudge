import torch
import torch.nn as nn

from net.model.encoder import LSTMEncoder
from net.model.decoder import LSTMArticleDecoder


class Article(nn.Module):
    def __init__(self, config, usegpu):
        super(Article, self).__init__()

        self.encoder = LSTMEncoder(config, usegpu)
        self.decoder = LSTMArticleDecoder(config, usegpu)
        self.dropout = nn.Dropout(config.getfloat("train", "dropout"))

    def init_hidden(self, config, usegpu):
        self.encoder.init_hidden(config, usegpu)
        self.decoder.init_hidden(config, usegpu)

    def forward(self, x, doc_len, config):
        x = self.encoder(x, doc_len, config)
        x = self.dropout(x)
        x = self.decoder(x, doc_len, config, self.encoder.attention)

        return x
