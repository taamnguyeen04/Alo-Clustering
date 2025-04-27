import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim=13, embedding_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, embedding_dim)
        )

    def forward(self, x):
        return self.model(x)

class ClusteringHead(nn.Module):
    def __init__(self, embedding_dim=64, num_clusters=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_clusters)  # logits
        )

    def forward(self, z):
        return self.model(z)
