"""
Dosovitskiy et al. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv:2010.11929
Vision transformer implementation.
"""

from torch import nn

from patches import EmbeddedPatches
from multihead_attention import MultiheadAttention
from multilayer_perceptron import MultilayerPerceptron


class TransformerEncoder(nn.Module):

    def __init__(self,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 hidden_dim: int = 512,
                 mlp_out: int = 256,
                 drop_rate: float = 0.):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = MultiheadAttention(embed_dim=embed_dim,
                                      num_heads=num_heads,
                                      drop_rate=drop_rate)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MultilayerPerceptron(embed_dim=embed_dim,
                                        hidden_dim=hidden_dim,
                                        output_dim=mlp_out,
                                        drop_rate=drop_rate)

    def forward(self, x):

        x1 = self.norm1(x)
        post_mha = self.mha(x1)

        x2 = x + post_mha
        x3 = self.norm2(x2)
        post_mlp = self.mlp(x3)

        encoder_output = post_mha + post_mlp

        return encoder_output


class VisionTransformer(nn.Module):

    def __init__(self,
                 img_size: int = 32,
                 patch_size: int = 4,
                 channels: int = 3,
                 num_classes: int = 10,
                 embed_dim: int = 256,
                 hidden_dim: int = 512,
                 mlp_out: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 drop_rate: float = 0.):
        super().__init__()

        self.embed_patches = EmbeddedPatches(img_size=img_size,
                                             patch_size=patch_size,
                                             channels=channels,
                                             embed_dim=embed_dim,
                                             drop_rate=drop_rate)

        self.transformer_encoder = nn.Sequential(
            *(TransformerEncoder(embed_dim=embed_dim,
                                 num_heads=num_heads,
                                 hidden_dim=hidden_dim,
                                 mlp_out=mlp_out,
                                 drop_rate=drop_rate) for _ in range(num_layers))
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.classification_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        patch_embed = self.embed_patches(x)
        transformer_out = self.transformer_encoder(patch_embed)
        cls = transformer_out[:, 0, :]
        out = self.classification_head(cls)

        return out

