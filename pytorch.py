import torch
import torch.nn as nn
import torch.optim as optim

print("PyTorch version: ", torch.__version__)
print("CUDA available: ", torch.cuda.is_available())

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

z = x + y

print("x = ", x)
print("y = ", y)
print("x + y = ", z)

layer = nn.Linear(3, 2)

output = layer(x)

print(output)

model = nn.Linear(1, 1)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

x = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[2.0], [4.0], [6.0]])

for epoch in range(100):
    prediction = model(x)

    loss = criterion(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Final loss: ", loss.item())