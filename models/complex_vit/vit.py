import torch
import torch.nn as nn
from models.complex_vit.complex_layers import ComplexLinear, ComplexLayerNorm
from models.complex_vit.patch_embedding import ComplexPatchEmbedding
from models.complex_vit.transformer_block import ComplexTransformerEncoderBlock


class ComplexReconstructionHead(nn.Module):
    def __init__(self, embed_dim, patch_size, image_size):
        super().__init__()
        assert image_size % patch_size == 0, \
            "Image size must be divisible by patch size"
        self.patch_size = patch_size
        self.image_size = image_size

        #Each token predicts a complex 16 x 16 patch
        self.linear_proj = ComplexLinear(
            embed_dim,
            patch_size ** 2
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.linear_proj(x)

        num_patches = self.image_size // self.patch_size

        #Unpatchify (B, 576, 256) -> (B, 384, 384), same ordering as the real head
        x = x.reshape(batch_size, num_patches, num_patches, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 3, 2, 4)
        x = x.reshape(batch_size, self.image_size, self.image_size)

        #Magnitude of the predicted complex image, compared against the real-valued RSS target
        return torch.abs(x)


class ComplexVisionTransformer(nn.Module):
    #embed_dim=138 / mlp_dim=552 matches the real ViT's parameter count (~3.28M vs ~3.22M real parameters).
    #Using the real ViT's 192 / 768 gives ~6.05M, since every complex weight is two real parameters
    def __init__(self, embed_dim=138, num_heads=6, in_channels=4, patch_size=16, mlp_dim=552, depth=6, image_size=384):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            "Embedding dimension must be divisible by number of heads"
        #Complex patch embedding (in_channels = 4 complex virtual coils)
        self.patch_embedding = ComplexPatchEmbedding(in_channels, embed_dim, patch_size, image_size)

        #Complex transformer encoder stack
        self.blocks = nn.Sequential(
            *[ComplexTransformerEncoderBlock(embed_dim, num_heads, mlp_dim) for _ in range(depth)]
        )
        #Final normalization
        self.norm = ComplexLayerNorm(embed_dim)
        #Reconstruction head (complex tokens -> real magnitude image)
        self.reconstruction_head = ComplexReconstructionHead(embed_dim, patch_size, image_size)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.reconstruction_head(x)
        return x
