import torch

target = torch.Tensor([0, 1, 0, 2, 3, 4, 0])
mask = target != torch.Tensor([0])

denom = torch.sum(mask, -1, keepdim=True)

y = torch.sum(target * mask) / denom

print(y)
