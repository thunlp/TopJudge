import torch
import random
import torch.utils.data
import torch.legacy as L


def init_loader(config):
    data = []
    for a in range(0, 1024 * 32):
        data.append((torch.rand(1, 100, 100), random.randint(0, 3)))
    loader = torch.utils.data.DataLoader(dataset=data, batch_size=config.getint("train", "batch_size"))

    return loader
