#Fine-grained pest model
import torch
from torch import nn
from torch import Tensor
from torchvision import models
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from ViT import TransformerEncoder
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.dim_k = 384
        self.linear_q = nn.Linear(1, self.dim_k)
        self.linear_k = nn.Linear(1, self.dim_k)
        self.linear_v = nn.Linear(1, self.dim_k)
        self.down = nn.Linear(self.dim_k, 1)
        self.scale = self.dim_k ** -0.5
    def forward(self, x): 
        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v
        weight = torch.softmax(torch.matmul(q,k.transpose(-1,-2))*self.scale, dim=-1)
        att = torch.matmul(weight,v)
        att = self.down(att)
        return att
class SEPlus(nn.Module):
    def __init__(self, in_channel):
        super(SEPlus, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.attention = Attention()
        self.sigmoid = nn.Sigmoid()
    def forward(self, inputs):
        b, c, h, w = inputs.shape
        # [b,c,h,w]==>[b,c,1,1]
        x = self.avg_pool(inputs)
        x = rearrange(x, "b c h w -> b c (h w)")
        x = self.attention(x)
        x = self.sigmoid(x)
        # [b,c]==>[b,c,1,1]
        x = x.view([b,c,1,1])
        outputs = x * inputs
        return outputs
class AdaptiveFeatureFiltering(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 32, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size)**2+1, emb_size))
        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x
class PestModel(nn.Module):
    def __init__(self, num_classes, dim = 1024, depth = 1):
        super().__init__()
        self.aff = AdaptiveFeatureFiltering(emb_size = dim)
        self.zerovector = nn.Parameter(torch.zeros(1,1,dim,requires_grad=True))
        self.bias = nn.Parameter(torch.Tensor([0.]))
        self.seplus = SEPlus(in_channel = dim)
        self.backbone = models.convnext_large(pretrained = True, cache_dir = './')
        #Delete the relevant code in pytorch library
        del self.backbone.classifier
        del self.backbone.avgpool
        self.encoder = TransformerEncoder(depth = depth, emb_size = dim, dropout = 0.1)
        self.global_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    def forward(self,x):
        b, _, _, _ = x.shape
        x_ = self.aff(x) # filtered features
        x = self.backbone(x) # torch.Size([1, 1536, 7, 7])
        temp = x
        x = self.seplus(x)
        x = x + temp
        x = rearrange(x, "b c h w -> b (h w) c")
        fill = repeat(self.zerovector,'() n e -> b n e', b=b).to(x.device) # torch.Size([b, 1, 1536])
        x = torch.cat([fill, x], dim=1) # torch.Size([1, 50, 1536])
        x = x + (1+self.bias)*x_
        x = self.encoder(x)
        x = rearrange(x, "b h c -> b c h")
        x = self.global_pool(x)
        x = x.squeeze(2)
        x = self.head(x)  
        return x
