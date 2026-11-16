import torch 
import math
import torch.nn as nn 

class MultiHeadedAttention(nn.Module): 
    def __init__(self, embed_dim, num_heads): 
        super().__init__()
        self.embed_dim = embed_dim 
        self.num_heads = num_heads 
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x): 
        batch_size, num_tokens, embed_dim = x.shape
        num_heads = self.num_heads 
        head_dim = self.head_dim

        query = self.query(x)
        query = query.reshape(batch_size, num_tokens, num_heads, head_dim)
        query = query.transpose(1,2)

        key = self.key(x)
        key = key.reshape(batch_size, num_tokens, num_heads, head_dim)
        key = key.transpose(1,2)

        value = self.value(x) 
        value = value.reshape(batch_size, num_tokens, num_heads, head_dim)
        value = value.transpose(1,2)

        query_key_pair = torch.matmul(query, key.transpose(-2, -1))
        attention_weights = torch.softmax((query_key_pair/math.sqrt(head_dim)), dim=-1)

        attention_output = torch.matmul(attention_weights, value)
        attention_output = attention_output.transpose(1, 2)
        attention_output = attention_output.reshape(batch_size, num_tokens, embed_dim)
        attention_output = self.output_proj(attention_output)

        return attention_output 

#testing output dims 
#model = MultiHeadedAttention(embed_dim=192, num_heads=6)
#x = torch.randn(4, 576, 192)
#output = model(x)
#print(output.shape)
#print(sum(p.numel() for p in model.parameters()))