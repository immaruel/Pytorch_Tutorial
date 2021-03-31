import torch


# x = torch.randn(3, requires_grad = True)
# print(x)

# 3가지방법 : gradient 못하게 하기
# 1. x.requires_grad_(False)
# 2. x.detach()
# 3. with torch.no_grad():


weights = torch.ones(4, requires_grad = True)
optimizer = torch.optim.SGD(weights, lr=0.01)
optimizer.step()
optimizer.zero_grad()