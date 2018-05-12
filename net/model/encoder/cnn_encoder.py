import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    def __init__(self, config, usegpu):
        super(CNNEncoder, self).__init__()

        self.convs = []
        for a in range(config.getint("net", "min_gram"), config.getint("net", "max_gram") + 1):
            self.convs.append(nn.Conv2d(1, config.getint("net", "filters"), (a, config.getint("data", "vec_size"))))

        self.convs = nn.ModuleList(self.convs)
        self.feature_len = (-config.getint("net", "min_gram") + config.getint("net", "max_gram") + 1) * config.getint(
            "net", "filters")

    def forward(self, x, doc_len, config):
        x = x.view(config.getint("data", "batch_size"), 1, -1, config.getint("data", "vec_size"))
        conv_out = []
        gram = config.getint("net", "min_gram")
        self.attention = []
        for conv in self.convs:
            y = F.relu(conv(x)).view(config.getint("data", "batch_size"), config.getint("net", "filters"), -1)
            self.attention.append(F.pad(y, (0, gram - 1)))
            # print("gg",type(x))
            y = F.max_pool1d(y, kernel_size=config.getint("data", "sentence_num") * config.getint("data",
                                                                                                  "sentence_len") - gram + 1).view(
                config.getint("data", "batch_size"), -1)
            # y =
            conv_out.append(y)
            gram += 1

        conv_out = torch.cat(conv_out, dim=1)

        self.attention = torch.cat(self.attention, dim=1)
        fc_input = conv_out

        features = (config.getint("net", "max_gram") - config.getint("net", "min_gram") + 1) * config.getint("net",
                                                                                                             "filters")

        fc_input = fc_input.view(-1, features)
        # print(fc_input)

        return fc_input
