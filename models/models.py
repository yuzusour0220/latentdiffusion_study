import torch
import torch.nn as nn

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels,t_emb_dim, num_layers, norm_channels):
        super().__init__()
        self.num_layers = num_layers
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

    def forward(self, x, t_emb=None):
        out = x
        for i in range(self.num_layers):
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input) 
            
            