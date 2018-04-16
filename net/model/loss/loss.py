import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_loss(outputs, labels):
    temp = outputs
    res = - labels * torch.log(temp) - (1 - labels) * torch.log(1 - temp)
    res = torch.mean(torch.sum(res, dim=1))

    return res
