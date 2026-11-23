import torch
import torch.nn as nn
from models.complex_vit.complex_attention import ComplexMultiHeadedAttention
from models.complex_vit.complex_layers import ComplexLinear, ComplexLayerNorm, ComplexGELU


class ComplexTransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        #Same pre-norm layout as the real block, with every layer replaced by its complex counterpart
        self.self_attention_norm = ComplexLayerNorm(embed_dim)
        self.ff_norm = ComplexLayerNorm(embed_dim)

        self.attention_block = ComplexMultiHeadedAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            #"Up projection"
            ComplexLinear(embed_dim, mlp_dim),
            ComplexGELU(),
            #"Down projection" back to embed_dim for the residual connection
            ComplexLinear(mlp_dim, embed_dim)
        )

    def forward(self, x):
        #layernorm -> attention -> residual -> layernorm -> ffn -> residual
        selfA_norm = self.self_attention_norm(x)
        attention_output = self.attention_block(selfA_norm)
        x = x + attention_output

        ff_norm = self.ff_norm(x)
        ffn_output = self.ffn(ff_norm)
        x = x + ffn_output
        return x
