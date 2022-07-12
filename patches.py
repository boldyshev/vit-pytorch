"""
Dosovitskiy et al. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv:2010.11929
Implementation of patch embeddings.
"""

import torch
from torch import nn


class EmbeddedPatches(nn.Module):
    """
    Using 2D convolutions to create patch embeddings instead of linear layer
    """

    def __init__(self,
                 img_size: int = 32,
                 patch_size: int = 4,
                 channels: int = 3,
                 embed_dim: int = 256,
                 drop_rate: float = 0.):

        super().__init__()
        #  Section 3.1, first paragraph, page 3 of the paper
        self.num_patches = (img_size // patch_size) ** 2

        # Equation 1
        self.patch_embedding = nn.Conv2d(channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # x_class, eq. 1
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # E_pos, eq. 1. num_patches + 1, considering cls_token
        self.pos_embedding = nn.Parameter(torch.randn(self.num_patches + 1, embed_dim))

        # Dropout after each dense layer, appendix B.1, page 13
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):

        # input image tensor has shape (batch_size, channels, height, width)
        batch_size = x.shape[0]

        # create patch embeddings:
        # (batch_size, channels, height, width) -> (batch_size, embed_dim, patch_size, patch_size)
        x = self.patch_embedding(x)

        # (batch_size, embed_dim, patch_size, patch_size) -> (batch_size, patch_size **2, embed_dim)
        x = x.flatten(2).permute(0, 2, 1)

        # expand single classification token to batch size
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Eq. 1
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding

        # Dropout after each dense layer, section B.1, page 13
        x = self.dropout(x)

        return x
