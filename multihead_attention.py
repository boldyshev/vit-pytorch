"""
Dosovitskiy et al. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv:2010.11929
Implementation of Multi-Head Self-Attention block.
"""

from torch import nn


class MultiheadAttention(nn.Module):

    def __init__(self,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 drop_rate: float = 0.):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        # matrices for query, key, value
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)

        # project back to input dimension
        self.linear = nn.Linear(self.embed_dim, self.embed_dim)

        # Dropout after each dense layer, appendix B.1, page 13
        # Different dropout objects for different layers according to
        # https://discuss.pytorch.org/t/using-same-dropout-object-for-multiple-drop-out-layers/39027/2
        self.attention_dropout = nn.Dropout(p=drop_rate)
        self.linear_dropout = nn.Dropout(p=drop_rate)

    def split_heads(self, x):
        """Split the last dimension into (num_heads, head_dim)
        """
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3)

        return x

    def forward(self, x):
        batch_size, patch_number, flat_dim = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, patch_number, 3, self.num_heads, flat_dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.attention_dropout(attention)

        x = attention @ v
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, patch_number, flat_dim)

        x = self.linear(x)
        x = self.linear_dropout(x)

        return x
