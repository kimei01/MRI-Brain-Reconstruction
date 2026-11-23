import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        #Real and imaginary parts are stored as separate real parameters, so parameter counts
        #and AdamW behave exactly like the real model (1 complex weight = 2 real parameters)
        self.weight_real = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias_real = nn.Parameter(torch.empty(out_features))
            self.bias_imag = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)
        self.reset_parameters()

    def reset_parameters(self):
        #nn.Linear default init has variance 1/(3 * fan_in); split it evenly between the real and
        #imaginary parts so E|w|^2 matches the real layer
        bound = 1 / math.sqrt(2 * self.in_features)
        for param in self.parameters():
            nn.init.uniform_(param, -bound, bound)

    def forward(self, x):
        #(Wr + iWi)(xr + ixi) = (Wr xr - Wi xi) + i(Wr xi + Wi xr)
        weight = torch.complex(self.weight_real, self.weight_imag)
        bias = None
        if self.bias_real is not None:
            bias = torch.complex(self.bias_real, self.bias_imag)
        return F.linear(x, weight, bias)


class ComplexLayerNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        #Complex affine parameters, initialized to the identity (scale 1 + 0i, shift 0 + 0i)
        self.weight_real = nn.Parameter(torch.ones(embed_dim))
        self.weight_imag = nn.Parameter(torch.zeros(embed_dim))
        self.bias_real = nn.Parameter(torch.zeros(embed_dim))
        self.bias_imag = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        #Subtract the complex mean, then divide by the complex standard deviation sqrt(E|z - mean|^2).
        #Scaling by a real number keeps the phase of every feature intact
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        variance = (centered.real ** 2 + centered.imag ** 2).mean(dim=-1, keepdim=True)
        x = centered * torch.rsqrt(variance + self.eps)

        weight = torch.complex(self.weight_real, self.weight_imag)
        bias = torch.complex(self.bias_real, self.bias_imag)
        return x * weight + bias


class ComplexGELU(nn.Module):
    #Split activation: GELU on the real and imaginary parts independently (complex counterpart of
    #the real ViT's GELU, same family as CReLU)
    def forward(self, x):
        return torch.complex(F.gelu(x.real), F.gelu(x.imag))
