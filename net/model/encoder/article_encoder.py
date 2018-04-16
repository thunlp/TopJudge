import torch
import torch.nn as nn
from net.model.encoder import CNNEncoder
from net.file_reader import generate_article_list


class ArticleEncoder(nn.Module):
    def __init__(self, config, usegpu):
        super(ArticleEncoder, self).__init__()

        self.article_encoder = CNNEncoder(config, usegpu)
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
