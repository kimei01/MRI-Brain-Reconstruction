import torch
import math
import torch.nn as nn
from models.complex_vit.complex_layers import ComplexLinear


class ComplexMultiHeadedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        #Complex QKV projections
        self.query = ComplexLinear(embed_dim, embed_dim)
        self.key = ComplexLinear(embed_dim, embed_dim)
        self.value = ComplexLinear(embed_dim, embed_dim)
        #Complex output projection
        self.output_proj = ComplexLinear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape
        num_heads = self.num_heads
        head_dim = self.head_dim

        #Reshape to (B, heads, tokens, head_dim) so attention runs independently per head
        query = self.query(x)
        query = query.reshape(batch_size, num_tokens, num_heads, head_dim)
        query = query.transpose(1, 2)

        key = self.key(x)
        key = key.reshape(batch_size, num_tokens, num_heads, head_dim)
        key = key.transpose(1, 2)

        value = self.value(x)
        value = value.reshape(batch_size, num_tokens, num_heads, head_dim)
        value = value.transpose(1, 2)

        #Softmax needs real scores. Use the real part of the Hermitian product:
        #Re(Q K^H) = Qr Kr^T + Qi Ki^T, which is unchanged if Q and K share a phase rotation
        query_key_pair = (
            torch.matmul(query.real, key.real.transpose(-2, -1))
            + torch.matmul(query.imag, key.imag.transpose(-2, -1))
        )
        attention_weights = torch.softmax((query_key_pair / math.sqrt(head_dim)), dim=-1)

        #Real weights mix the real and imaginary parts of V identically, so the phase of V is preserved
        attention_output = torch.complex(
            torch.matmul(attention_weights, value.real),
            torch.matmul(attention_weights, value.imag)
        )
        attention_output = attention_output.transpose(1, 2)
        attention_output = attention_output.reshape(batch_size, num_tokens, embed_dim)
        attention_output = self.output_proj(attention_output)

        return attention_output
