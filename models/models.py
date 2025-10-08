import torch
import torch.nn as nn
from einops import einsum

def PositionEmbeddings(time_steps, temb_dim):
    # 割り方を変えた。エラーが出るかも。
    factor = 10000 ** (torch.arange(start=0, end=temb_dim//2, dtype=torch.float32, device=time_steps.device) * 2 / temb_dim)
    # まずはtime_stepsを縦ベクトルに変換。その後、それを横に並べる感じ。
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads, num_layers, attn, norm_channels):
        super().__init__()
        self.num_layers = num_layers
        self.attn = attn
        self.resnet_conv_first = nn.ModuleList(
            nn.Sequential(
                nn.GroupNorm(num_groups=norm_channels, num_channels=in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            ) for i in range(self.num_layers)
        )
        self.t_emb_dim = t_emb_dim
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList(
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(self.t_emb_dim, out_channels)
                ) for _ in range(self.num_layers)
            )
        self.resnet_conv_second = nn.ModuleList(
            nn.Sequential(
                nn.GroupNorm(num_groups=norm_channels, num_channels=out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            ) for _ in range(self.num_layers)
        )
        # 次元を揃えて足し算できるようにするための1x1畳み込み
        self.residual_input_conv = nn.ModuleList(
            [nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1) for i in range(self.num_layers)]
        )
        if self.attn:
            self.attention_norms = nn.ModuleList(
                [nn.GroupNorm(num_groups=norm_channels, num_channels=out_channels) for _ in range(self.num_layers)]
            )
            self.attentions = nn.ModuleList(
                [nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True) for _ in range(self.num_layers)]
            )

    def forward(self, x, t_emb=None):
        out = x
        for i in range(self.num_layers):
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input) 
            # shape: (batchsize, channels, height, width)
            if self.attn:
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.permute(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn
            return out

                
                

class Unet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.t_proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )
        
    def forward(self, x, t, cond_input=None):
        out = self.conv_in(x)
        # .long()で整数に変換
        t_emb = PositionEmbeddings(torch.as_tensor(t).long(), 256)
        t_emb = self.t_proj(t_emb)
        # class_embed = cond_input @ self.class_emb.weightと等価
        class_embed = einsum(cond_input.float(), self.class_emb.weight, "b n, n d -> b d")
        t_emb = t_emb + class_embed
        down_outs = []
