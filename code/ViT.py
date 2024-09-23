# -*- coding: utf-8 -*-
"""
Created on Fri May 12 13:43:18 2023

@author: HP
"""
import torch
import torch.nn.functional as F
#import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
#import torchsummary
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # 使用单个矩阵一次性计算出queries,keys,values
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # 将queries，keys和values划分为num_heads
        #print("1qkv's shape: ", self.qkv(x).shape)  # 使用单个矩阵一次性计算出queries,keys,values
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)  # 划分到num_heads个头上
        #print("2qkv's shape: ", qkv.shape)
        
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        #print("queries's shape: ", queries.shape)
        #print("keys's shape: ", keys.shape)
        #print("values's shape: ", values.shape)
        
        # 在最后一个维度上相加
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        #print("energy's shape: ", energy.shape)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        
        scaling = self.emb_size ** (1/2)
        #print("scaling: ", scaling)
        att = F.softmax(energy / scaling, dim=-1) 
        #print("att1' shape: ", att.shape)
        att = self.att_drop(att)
        #print("att2' shape: ", att.shape)
        
        # 在第三个维度上相加
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        #print("out1's shape: ", out.shape)
        out = rearrange(out, "b h n d -> b n (h d)")
        #print("out2's shape: ", out.shape)
        out = self.projection(out)
        #print("out3's shape: ", out.shape)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))
        
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes))

