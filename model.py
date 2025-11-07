import torch.nn as nn

class SimpleSurvModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleSurvModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)
