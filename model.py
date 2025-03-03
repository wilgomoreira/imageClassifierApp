import torch
import torch.nn as nn

class MLPNN(nn.Module):
    def __init__(self, input_dim, num_neurons=128, num_classes=1):
        super(MLPNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, num_neurons)
        self.fc2 = nn.Linear(num_neurons, num_neurons)
        self.fc3 = nn.Linear(num_neurons, num_classes)
    
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

