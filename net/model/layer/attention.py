import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        pass

    def forward(self, feature, hidden):
        feature = feature.view(feature.size(0), -1, 1)
        ratio = torch.bmm(hidden, feature)
        ratio = ratio.view(ratio.size(0), ratio.size(1))
        ratio = F.softmax(ratio, dim=1).view(ratio.size(0), -1, 1)
        result = torch.bmm(hidden.transpose(1, 2), ratio)
        result = result.view(result.size(0), -1)

        return result
