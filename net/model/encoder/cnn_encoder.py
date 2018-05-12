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
        for conv in self.convs:
            # print("gg",type(x))
            y = F.max_pool1d(F.relu(conv(x)).view(config.getint("data", "batch_size"), config.getint("net", "filters"),-1),
                             kernel_size=config.getint("data", "sentence_num") * config.getint("data",
                                                                                               "sentence_len") - gram + 1).view(config.getint("data","batch_size"),-1)
            # y = F.pad(y,
            #          (0, config.getint("data", "sentence_num") * config.getint("data", "sentence_len") - len(y[0][0])))
            conv_out.append(y)
            gram += 1

        conv_out = torch.cat(conv_out, dim=1)
        # conv_out = conv_out.view(config.getint("data", "batch_size"), -1,
        #                         config.getint("data", "sentence_num") * config.getint("data", "sentence_len"))

        #self.attention = conv_out.transpose(1, 2)
        # print(conv_out)
        fc_input = torch.max(conv_out, dim=2)[0]
        # print(fc_input)

        features = (config.getint("net", "max_gram") - config.getint("net", "min_gram") + 1) * config.getint("net",
                                                                                                             "filters")

        fc_input = fc_input.view(-1, features)
        # print(fc_input)

        return fc_input
