import torch 
import torch.nn as nn
from .real_attention import MultiHeadedAttention

class TransformerEncoderblock(nn.Module): 
    def __init__(self, embed_dim, num_heads,mlp_dim): 
        #learnable matrices
        #learns parameters entering attention
        self.self_attention_norm = nn.LayerNorm(embed_dim)
        #learns parameters entering mlp(multilayerperceptron)
        self.ff_norm = nn.LayerNorm(embed_dim)

        self.attention_block = MultiHeadedAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            #"Up projection" expands the linear layer 
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            #"Down projection" shrinks down the layer to match inputs dims from residual connections 
            nn.Linear(mlp_dim, embed_dim)
        )

    def forward(self, x):
        #In theory, it should go layernorm->attention output-> residual -> layernorm -> ffn-> residual and this cycle repeats. this is for each attention head 
        selfA_norm = self.self_attention_norm(x)
        attention_output = self.attention_block(selfA_norm)
        x = x + attention_output

        ff_norm = self.ff_norm(x)
        ffn_output = self.ffn(ff_norm)
        x = x + ffn_output
        return x

       