import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        #Convert input_dims into expected patch_dims 
        self.projection = nn.Conv2d(
            #in channels per slice 
            in_channels,
            #embedding dimensions (can be tested further to see performance tradeoff, i picked this for my own machine)
            embed_dim,
            #16 x 16 patch 
            kernel_size=patch_size,
            stride=patch_size
            #Output (B, 192, 24, 24)
        )

        self.num_patches = (384 // patch_size) ** 2
        # Add learnable positional information to each patch token
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
            )

    def forward(self, x):
        #Get patch dimensions
        x = self.projection(x)
        #Flatten 24 x 24 to tokens
        x = x.flatten(2)
        #Move tokens to expected position for transformer input (B, 576, 192)
        x = x.transpose(1, 2)
        x = x + self.pos_embedding

        return x
