import torch
import torch.nn as nn

T_IN, T_OUT = 10, 30
C = 64                 # VN channels per point (our choice)
D = C * 3              # token size: 64 vectors x 3 coordinates = 192


class VNLinear(nn.Module):
    """(..., c_in, 3) -> (..., c_out, 3). Mixes channels, never x/y/z."""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.lin = nn.Linear(c_in, c_out, bias=False)

    def forward(self, x):
        return self.lin(x.transpose(-1, -2)).transpose(-1, -2)


class VNReLU(nn.Module):
    """Rotation-safe ReLU: cuts each vector against a learned direction."""
    def __init__(self, c):
        super().__init__()
        self.dir = VNLinear(c, c)

    def forward(self, x):
        k = self.dir(x)
        dot = (x * k).sum(-1, keepdim=True)
        k2 = (k * k).sum(-1, keepdim=True) + 1e-6
        return torch.where(dot >= 0, x, x - dot / k2 * k)


class Policy(nn.Module):
    def __init__(self, layers=4, heads=4):
        super().__init__()
        self.encode = nn.Sequential(VNLinear(T_IN, C), VNReLU(C), VNLinear(C, C))
        self.tip_pos = nn.Parameter(torch.zeros(5, D))     # learned, fingertips only
        layer = nn.TransformerEncoderLayer(D, heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.Linear(D, 256), nn.ReLU(), nn.Linear(256, T_OUT * 3))

    def forward(self, tips, obj):
        """tips (B,10,5,3), obj (B,10,500,3) -> (B,30,5,3)"""
        B = tips.shape[0]                                   # batch size
        x = torch.cat([tips, obj], dim=2)                   # (B,10,505,3)  fingertips first
        x = x.permute(0, 2, 1, 3)                           # (B,505,10,3)  each point: its history
        x = self.encode(x)                                  # (B,505,64,3)  Vector Neuron MLP
        x = x.reshape(B, -1, D)                             # (B,505,192)   one token per point
        x = torch.cat([x[:, :5] + self.tip_pos, x[:, 5:]], dim=1)   # code on fingertips only
        x = self.transformer(x)                             # (B,505,192)
        y = self.head(x[:, :5])                             # (B,5,90)      fingertip tokens only
        y = y.reshape(B, 5, T_OUT, 3)                       # (B,5,30,3)
        return y.permute(0, 2, 1, 3)                        # (B,30,5,3)


if __name__ == "__main__":
    model = Policy()
    out = model(torch.randn(2, 10, 5, 3), torch.randn(2, 10, 500, 3))
    print(out.shape)      # expect torch.Size([2, 30, 5, 3])