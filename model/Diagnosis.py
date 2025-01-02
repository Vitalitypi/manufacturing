import torch.nn as nn
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(6, 128)  # 输入层到隐藏层
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)  # 隐藏层到隐藏层
        self.fc3 = nn.Linear(64, 7)  # 隐藏层到输出层

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out