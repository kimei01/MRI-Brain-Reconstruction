import torch 
import torch.nn as nn
from models.real_vit.patch_embedding import PatchEmbedding
from models.real_vit.transformer_block import TransformerEncoderBlock
class ReconstructionHead(nn.Module):
    def __init__(self, embed_dim, patch_size, image_size):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size

        self.linear_proj = nn.Linear(
            embed_dim,
            patch_size ** 2
        )
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.linear_proj(x)
        
        assert self.image_size % self.patch_size == 0, \
            "Image size must be divisible by patch size"
        num_patches = self.image_size // self.patch_size

        x = x.reshape(batch_size, num_patches, num_patches, self.patch_size, self.patch_size) 
        x = x.permute(0, 1, 3, 2, 4)
        x = x.reshape(batch_size, self.image_size, self.image_size)
        return x 
    
class RealVisionTransformer(nn.Module): 
    def __init__(self, embed_dim=192, num_heads=6,in_channels=8,patch_size=16,mlp_dim=768,depth=6,image_size = 384): 
        super().__init__()
        assert embed_dim % num_heads == 0, \
            "Embedding dimension must be divisible by number of heads"
        # Patch embedding
        self.patch_embedding = PatchEmbedding(in_channels, embed_dim, patch_size)

        # Transformer encoder stack
        self.blocks = nn.Sequential(
            *[TransformerEncoderBlock(embed_dim, num_heads, mlp_dim) for _ in range(depth)]
        )
        # Final normalization 
        self.norm = nn.LayerNorm(embed_dim)
        # Reconstruction head
        self.reconstruction_head = ReconstructionHead(embed_dim, patch_size, image_size)

    def forward(self,x): 
        x = self.patch_embedding(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.reconstruction_head(x)
        return x 