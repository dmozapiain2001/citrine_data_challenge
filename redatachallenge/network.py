import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.layers = nn.Sequential(
            nn.Linear(98, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 9)
        )
    def forward(self, batch):
        return self.layers(batch)
