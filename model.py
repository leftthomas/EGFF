import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, vgg16


class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial BCHW tensors """

    def __init__(self, num_channels):
        super().__init__(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)


class GlobalContext(nn.Module):
    def __init__(self, channel, reduction=4):
        super(GlobalContext, self).__init__()
        self.conv_attn = nn.Conv2d(channel, 1, 1, bias=True)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=True),
            LayerNorm2d(channel // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=True))

    def forward(self, x):
        b, c, _, _ = x.shape
        attn = self.conv_attn(x).reshape(b, 1, -1)
        attn = F.softmax(attn, dim=-1).unsqueeze(3)
        context = x.reshape(b, c, -1).unsqueeze(1) @ attn
        context = context.view(b, c, 1, 1)
        y = torch.sigmoid(self.fc(context))
        return x * y


class CBAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CBAM, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False))

    def forward(self, x):
        x_avg = self.fc(F.adaptive_avg_pool2d(x, (1, 1)))
        x_max = self.fc(F.adaptive_max_pool2d(x, (1, 1)))
        y = torch.sigmoid(x_avg + x_max)
        return x * y


class Model(nn.Module):
    def __init__(self, backbone_type, proj_dim):
        super(Model, self).__init__()
        # encoder
        if backbone_type == 'resnet50':
            backbone = []
            for module in resnet50(pretrained=True).children():
                if not isinstance(module, (nn.Linear, nn.AdaptiveAvgPool2d)):
                    backbone.append(module)
            backbone = nn.Sequential(*backbone)
            self.feat = backbone[:5]
            self.common = backbone[5:]
        elif backbone_type == 'vgg16':
            backbone = vgg16(pretrained=True).features
            self.feat = backbone[:16]
            self.common = backbone[16:-1]
        else:
            raise NotImplementedError('Not support {} as backbone'.format(backbone_type))
        # atte and proj
        self.low_atte = CBAM(256)
        self.middle_atte = CBAM(512)
        self.high_atte = CBAM(2048 if backbone_type == 'resnet50' else 512)
        self.proj = nn.Linear(2048 if backbone_type == 'resnet50' else 512, proj_dim)
        self.backbone_type = backbone_type

    def forward(self, img):
        low_feat = self.feat(img)
        # low_feat = self.low_atte(low_feat)
        middle_feat = self.common[:1 if self.backbone_type == 'resnet50' else 7](low_feat)
        # middle_feat = self.middle_atte(middle_feat)
        high_feat = self.common[1 if self.backbone_type == 'resnet50' else 7:](middle_feat)
        # high_feat = self.high_atte(high_feat)

        feat = torch.flatten(F.adaptive_max_pool2d(high_feat, (1, 1)), start_dim=1)
        proj = self.proj(feat)
        return F.normalize(proj, dim=-1)
