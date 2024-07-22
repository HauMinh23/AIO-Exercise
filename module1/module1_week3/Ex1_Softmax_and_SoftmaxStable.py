# Homework1
import torch            # type: ignore
import torch.nn as nn  # type: ignore

# Change the data type to float
data = torch.tensor([1, 2, 3], dtype=torch.float)

Softmax_function = nn.Softmax(dim=0)
output = Softmax_function(data)


class MySoftmax(nn.Module):  # Softmax
    def __init__(self):
        super().__init__()

    def forward(self, x):
        torch.exp(x)
        x_exp = torch.exp(x)
        total = x_exp.sum(0, keepdim=True)
        return x_exp/total


my_softmax = MySoftmax()
output = my_softmax(data)


class MyStableSoftmax(nn.Module):   # Stable softmax
    def __init__(self):
        super().__init__()

    def forward(self, x):
        torch.max(data, dim=0, keepdim=True).values
        c = torch.max(x, dim=0)
        torch.exp(x)
        x_exp = torch.exp(x - c.values)
        total = x_exp.sum(0, keepdim=True)
        return x_exp / total


mystablesoftmax = MyStableSoftmax()
output = mystablesoftmax(data)
