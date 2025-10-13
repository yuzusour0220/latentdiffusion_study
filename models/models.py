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
    def __init__(self, in_channels, out_channels, t_emb_dim, down_sample, num_heads, num_layers, attn, norm_channels):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.attn = attn
        self.resnet_conv_first = nn.ModuleList(
            nn.Sequential(
                nn.GroupNorm(num_groups=norm_channels, num_channels=in_channels if i == 0 else out_channels),
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
class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads, num_layers, norm_channels):
        super().__init__()
        self.num_layers = num_layers
        self.t_emb_dim = t_emb_dim
        self.resnet_conv_first = nn.ModuleList(
            nn.Sequential(
                nn.GroupNorm(num_groups=norm_channels, num_channels=in_channels if i == 0 else out_channels),
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
            [nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1) for i in range(self.num_layers + 1)]
        )
        self.attention_norms = nn.ModuleList(
            [nn.GroupNorm(num_groups=norm_channels, num_channels=out_channels) for _ in range(self.num_layers)]
        )
        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True) for _ in range(self.num_layers)]
        )

    def forward(self, x, t_emb=None):
        out = x
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        if self.t_emb_dim is not None:
            out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)
        
        for i in range(self.num_layers):
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.permute(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn 
            
            resnet_input = out
            out = self.resnet_conv_second[i + 1](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i + 1](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i + 1](out)
            out = out + self.residual_input_conv[i + 1](resnet_input)
        return out
    

    
    
    
    
    
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample, num_heads, num_layers, attn, norm_channels, model_type):
        super().__init__()
        self.num_layers = num_layers
        self.attn = attn
        self.up_sample = up_sample
        self.resnet_conv_first = nn.ModuleList(
            nn.Sequential(
                nn.GroupNorm(num_groups=norm_channels, num_channels=in_channels if i == 0 else out_channels),
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
            
        # VAEのときは転置畳み込み、UNetのときはアップサンプリング＋畳み込み
        
        if self.model_type == 'unet':
            # ダウンサンプリング側の出力を足して２倍になったインチャネル数をもとに戻すための//2と、アップサンプリングするための//2
            self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 4, 2, 1) if self.up_sample else nn.Identity()
        else:
            self.up_sample_conv = nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1) if self.up_sample else nn.Identity()

    def forward(self, x, out_down, t_emb=None):
        x = self.up_sample_conv(x)
        if out_down is not None:
            x = torch.cat([x, out_down], dim=1)  # チャンネル方向に結合
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
        self.downs = nn.ModuleList([])
        self.downs.append(DownBlock(128, 256, t_emb_dim=256, down_sample=False, num_heads=16, num_layers=2, attn=True, norm_channels=32))
        self.downs.append(DownBlock(256, 256, t_emb_dim=256, down_sample=False, num_heads=16, num_layers=2, attn=True, norm_channels=32))
        self.downs.append(DownBlock(256, 256, t_emb_dim=256, down_sample=False, num_heads=16, num_layers=2, attn=True, norm_channels=32))
        
        self.mids = MidBlock(256, 256, t_emb_dim=256, num_heads=16, num_layers=2, norm_channels=32)
        
        self.ups = nn.ModuleList([])
        self.ups.append(UpBlock(256 * 2, 256, t_emb_dim=256, up_sample=False, num_heads=16, num_layers=2, attn=True, norm_channels=32, model_type='unet'))
        self.ups.append(UpBlock(256 * 2, 256, t_emb_dim=256, up_sample=False, num_heads=16, num_layers=2, attn=True, norm_channels=32, model_type='unet'))
        self.ups.append(UpBlock(256 * 2, 128, t_emb_dim=256, up_sample=False, num_heads=16, num_layers=2, attn=True, norm_channels=32, model_type='unet'))
        self.norm_out = nn.GroupNorm(32, 128)
        self.conv_out = nn.Conv2d(128, in_channels, kernel_size=3, padding=1)


    def forward(self, x, t, cond_input=None):
        out = self.conv_in(x)
        # .long()で整数に変換
        t_emb = PositionEmbeddings(torch.as_tensor(t).long(), 256)
        t_emb = self.t_proj(t_emb)
        # class_embed = cond_input @ self.class_emb.weightと等価
        class_embed = einsum(cond_input.float(), self.class_emb.weight, "b n, n d -> b d")
        t_emb = t_emb + class_embed
        down_outs = []
        for i, down in enumerate(self.downs):
            down_outs.append(out)
            out = down(out, t_emb)

        out = self.mids(out, t_emb)
        
        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb)
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        return out


class VQVAE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encoder_conv_in = nn.Conv2d(in_channels, 32, kernel_size=3, padding=(1, 1))
        self.encoder_layers = nn.ModuleList([])
        self.encoder_layers.append(DownBlock(32, 64, t_emb_dim=None, down_sample=False, num_heads=16, num_layers=1, attn=False, norm_channels=32))
        self.encoder_layers.append(DownBlock(64, 128, t_emb_dim=None, down_sample=True, num_heads=16, num_layers=1, attn=False, norm_channels=32))
        self.encoder_mids = MidBlock(128, 128, t_emb_dim=None, num_heads=16, num_layers=1, norm_channels=32)
        self.encoder_norm_out = nn.GroupNorm(32, 128)
        self.encoder_conv_out = nn.Conv2d(128, 3, kernel_size=3, padding=(1, 1))
        self.pre_quant_conv = nn.Conv2d(3, 3, kernel_size=1)
        self.embedding = nn.Embedding(20, 3) # 20はコードブックのサイズ、3は潜在ベクトルの次元数
        self.post_quant_conv = nn.Conv2d(3, 3, kernel_size=1)
        self.decoder_conv_in = nn.Conv2d(3, 128, kernel_size=3, padding=(1, 1))
        self.decoder_mids = MidBlock(128, 128, t_emb_dim=None, num_heads=16, num_layers=1, norm_channels=32)
        self.decoder_layers = nn.ModuleList([])
        self.decoder_layers.append(UpBlock(128, 64, t_emb_dim=None, up_sample=True, num_heads=16, num_layers=1, attn=False, norm_channels=32, model_type='vqvae'))
        self.decoder_layers.append(UpBlock(64, 32, t_emb_dim=None, up_sample=True, num_heads=16, num_layers=1, attn=False, norm_channels=32, model_type='vqvae'))
        self.decoder_norm_out = nn.GroupNorm(32, 32)
        self.decoder_conv_out = nn.Conv2d(32, in_channels, kernel_size=3, padding=(1, 1))

    def quantize(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.size(0), -1, x.size(-1))  # (B, H*W, C)
        # (K, C) → (1, K, C) → (B, K, C)して、xとコードブックのすべてのペアのユークリッド距離を計算
        dist = torch.cdist(x, self.embedding.weight[None, :].repeat((x.size(0), 1, 1)))
        min_encoding_indices = torch.argmin(dist, dim=-1)  # (B, H*W)
        # もっとも近いコードブックを選択 全位置分取ってくる感じ。
        quant_out = torch.index_select(self.embedding.weight, 0, min_encoding_indices.view(-1))
        x = x.reshape(-1, x.size(-1))
        commitment_loss = torch.mean((quant_out.detach() - x) ** 2)
        codebook_loss = torch.mean((quant_out - x.detach()) ** 2)
        quantize_losses = {
            'codebook_loss': codebook_loss,
            'commitment_loss': commitment_loss,
        }
        # 「順伝播では量子化した値（quant_out）を使い、逆伝播では元の連続値（x）を通すことにして、あたかも連続的に微分できるようにする」
        quant_out = x + (quant_out - x).detach()
        quant_out = quant_out.reshape((B, H, W, C)).permute(0, 3, 1, 2)
        min_encoding_indices = min_encoding_indices.reshape((-1, quant_out.size(-2), quant_out.size(-1)))  # (B, H, W)
        return quant_out, quantize_losses, min_encoding_indices
        
        

    def encode(self, x):
        out = self.encoder_conv_in(x)
        for down in self.encoder_layers:
            out = down(out)
        out = self.encoder_mids(out)
        out = self.encoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.encoder_conv_out(out)
        out = self.pre_quant_conv(out)
        out, quant_losses = self.quantize(out)
        return out, quant_losses

    
    def decode(self, z):
        out = z
        out = self.post_quant_conv(out)
        out = self.decoder_conv_in(out)
        out = self.decoder_mids(out)
        for up in self.decoder_layers:
            out = up(out)
        out = self.decoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.decoder_conv_out(out)
        return out
        
    def forward(self, x):
        z, quant_losses = self.encode(x)
        out = self.decode(z)
        return out, z, quant_losses
