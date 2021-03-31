import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0,requires_grad=True)

# 순전파
y_hat = w * x
loss = (y_hat - y)**2 
print(loss)

# 역전파
loss.backward()
print(w.grad)

# weight 업데이트
# 다음 순전파와 역전파


print(x)
print(y)