import torch
import torch.nn as nn
from models.complex_vit.complex_layers import ComplexLinear


class ComplexPatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, image_size=384):
        super().__init__()
        assert image_size % patch_size == 0, \
            "Image size must be divisible by patch size"
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2

        #Patchify + complex linear projection. Equivalent to the real model's Conv2d with
        #kernel_size = stride = patch_size, but works on complex tensors
        self.projection = ComplexLinear(in_channels * patch_size ** 2, embed_dim)

        #Add learnable complex positional information to each patch token
        self.pos_embedding_real = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_embedding_imag = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

    def forward(self, x):
        if not torch.is_complex(x):
            raise TypeError("ComplexPatchEmbedding expects a complex-valued input tensor (use x_complex).")
        batch_size, channels, height, width = x.shape
        assert height == self.image_size and width == self.image_size, \
            f"Expected {self.image_size}x{self.image_size} input, got {height}x{width}"
        patch_size = self.patch_size
        grid_size = self.image_size // patch_size

        #(B, C, 384, 384) -> (B, C, 24, 16, 24, 16)
        x = x.reshape(batch_size, channels, grid_size, patch_size, grid_size, patch_size)
        #Group pixels by patch, channel-major inside each patch like Conv2d: (B, 24, 24, C, 16, 16)
        x = x.permute(0, 2, 4, 1, 3, 5)
        #Flatten to tokens (B, 576, C * 16 * 16)
        x = x.reshape(batch_size, self.num_patches, channels * patch_size ** 2)

        #Project to complex token embeddings (B, 576, embed_dim)
        x = self.projection(x)
        x = x + torch.complex(self.pos_embedding_real, self.pos_embedding_imag)

        return x
