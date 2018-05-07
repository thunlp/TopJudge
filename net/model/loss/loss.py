import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_loss(outputs, labels):
    temp = F.sigmoid(outputs)
    res = - labels * torch.log(temp) - (1 - labels) * torch.log(1 - temp)
    res = torch.mean(torch.sum(res, dim=1))

    return res


def one_cross_entropy_loss(outputs, labels):
    criterion = nn.CrossEntropyLoss()
    return criterion(outputs, torch.max(labels,dim=1)[1])


def log_regression(outputs, labels):
    return torch.log(torch, abs(outputs - labels) + 1.0) ** 2
