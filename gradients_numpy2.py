
# 1) design model(input, output size, forward pass)
# 2) construct loss and optimizer
# 3) training loop
# - forward pass : compute predition
# - backward pass : gradients
# - update weights

import torch
import torch.nn as nn
# f = w * x

# f = 2 * x
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

#w = torch.tensor(0.0, dtype=torch.float32, requires_grad = True)
X_test = torch.tensor([5], dtype= torch.float32)
n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features

#model = nn.Linear(input_size, output_size) # input_size와 output_size -> model의 parameter

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # layer 정의
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)

# model prediction
#def forward(x):
#    return w * x


# loss -> MSE 사용
#def loss(y, y_predicted):
#    return ((y_predicted-y)**2).mean()


print(f'Training before prediction : f(5) = {model(X_test).item():.3f}')

# Training
learning_rate = 0.01
n_iters = 100

loss =nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

for epoch in range(n_iters):
    # prediction = foward pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward() # dl/dw

    # update weights
    #with torch.no_grad():
    #    w -= learning_rate * w.grad
    optimizer.step()

    # zero gradient
    # w.grad.zero_()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w,b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')
    
    print(f'Training before prediction : f(5) = {model(X_test).item():.3f}')
        