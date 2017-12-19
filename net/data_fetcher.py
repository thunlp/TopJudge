import torch
import random
import torch.utils.data

data = []
for a in range(0,1024):
    data.append(torch.rand(1, 1, random.randint(1, 100), 100))

loader = torch.utils.data.DataLoader(dataset=data,batch_size=16)

for batch in loader:
    print(batch)