#Fine-grained pest model
import torch
from torch import nn
from torch import Tensor
from torchvision import models
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from timm.models import create_model
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
        B, _, C = x.shape
        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v
        weight = torch.softmax(torch.matmul(q,k.transpose(-1,-2))*self.scale, dim=-1)
        att = torch.matmul(weight,v)
        att = self.down(att)
        return att
class SEBlock(nn.Module):
    # 初始化, in_channel代表输入特征图的通道数, ratio代表第一个全连接下降通道的倍数
    def __init__(self, in_channel, ratio=4):
        # 继承父类初始化方法
        super(SEBlock, self).__init__()
        
        # 属性分配
        # 全局平均池化，输出的特征图的宽高=1
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        #改进
        self.attention = Attention()
        self.sigmoid = nn.Sigmoid()
    def forward(self, inputs):  # inputs 代表输入特征图
        # 获取输入特征图的shape
        b, c, h, w = inputs.shape
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        x = self.avg_pool(inputs)
        x = rearrange(x, "b c h w -> b c (h w)")
        x = self.attention(x)
        # 对通道权重归一化处理
        x = self.sigmoid(x)
        # 调整维度 [b,c]==>[b,c,1,1]
        x = x.view([b,c,1,1])
        # 将输入特征图和通道权重相乘
        outputs = x * inputs
        return outputs
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 32, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        # 位置编码信息，一共有(img_size // patch_size)**2 + 1(cls token)个位置向量
        self.positions = nn.Parameter(torch.randn((img_size // patch_size)**2+1, emb_size))
        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x
#两个模块
class PestModel(nn.Module):
    def __init__(self, model_name,
        pretrained=False,
        pretrained_cfg=None,
        checkpoint_path='',
        scriptable=None,
        exportable=None,
        no_jit=None,
        **kwargs):
        super().__init__()
        dim = 1024
        depth=1
        self.patch_embedding = PatchEmbedding(emb_size = dim)
        self.zerovector = nn.Parameter(torch.zeros(1,1,dim,requires_grad=True))
        self.bias = nn.Parameter(torch.Tensor([0.]))
        self.senet = SEBlock(in_channel = dim)
        self.backbone = create_model(
            model_name = model_name,
            pretrained = pretrained,
            pretrained_cfg = pretrained_cfg,
            checkpoint_path = checkpoint_path,
            scriptable = scriptable,
            exportable = exportable,
            no_jit = no_jit,
            **kwargs
        )
        del self.backbone.head
        self.encoder = TransformerEncoder(depth = depth, emb_size = dim, dropout = 0.1)
        self.global_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 102)
        )
    def forward(self,x):
        b, _, _, _ = x.shape
        embeddings = self.patch_embedding(x)
        x = self.backbone(x) # torch.Size([1, 1024, 7, 7])
        #x = rearrange(x, "b h w c -> b c h w")
        temp = x
        x = self.senet(x)
        x = x + temp
        x = rearrange(x, "b c h w -> b (h w) c")
        fill = repeat(self.zerovector,'() n e -> b n e', b=b).to(x.device) # torch.Size([b, 1, 1024])
        x = torch.cat([fill, x], dim=1) # torch.Size([1, 50, 1024])
        x = x + (1+self.bias)*embeddings
        x = self.encoder(x)
        x = rearrange(x, "b h c -> b c h")
        x = self.global_pool(x)
        x = x.squeeze(2)
        x = self.head(x)
        return x
