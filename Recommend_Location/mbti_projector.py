import torch.nn as nn

class MBTIProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 768)
        )

    def forward(self, x):
        return self.net(x)