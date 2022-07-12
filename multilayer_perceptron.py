"""
Dosovitskiy et al. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv:2010.11929
Implementation of Multi-Layer Perceptron block.
"""

from torch import nn


class MultilayerPerceptron(nn.Module):

    def __init__(self,
                 embed_dim: int = 256,
                 hidden_dim: int = 512,
                 output_dim: int = 256,
                 drop_rate: float = 0.):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(p=drop_rate)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.dropout2 = nn.Dropout(p=drop_rate)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        
        return x
