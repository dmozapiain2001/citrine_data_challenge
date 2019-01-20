import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.layers = nn.Sequential(
            nn.Linear(98, 120),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(120, 60),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(60, 30),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(30, 9)
        )
    def forward(self, batch):
        return self.layers(batch)
