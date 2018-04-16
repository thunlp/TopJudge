import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionTanH(nn.Module):
    def __init__(self, config):
        super(AttentionTanH, self).__init__()
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
