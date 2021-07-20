import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import sobel
from torchvision.models import resnet50, vgg16


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


class GlobalContext(nn.Module):
    def __init__(self, channel, reduction=16):
        super(GlobalContext, self).__init__()
        self.cbam = CBAM(channel, reduction)
        self.attn = nn.Conv2d(3, 1, 1, bias=True)

    def forward(self, x, edge):
        edge = F.interpolate(edge, size=x.size()[-2:], mode='bilinear', align_corners=False)
        attn = torch.sigmoid(self.attn(edge))
        context = attn * self.cbam(x)
        return context


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
        self.low_atte = GlobalContext(256)
        self.middle_atte = GlobalContext(512)
        self.high_atte = GlobalContext(2048 if backbone_type == 'resnet50' else 512)
        self.proj = nn.Linear(256 + 512 + 2048 if backbone_type == 'resnet50' else 256 + 512 + 512, proj_dim)
        self.backbone_type = backbone_type

    def forward(self, img):
        edge = sobel(img)
        low_feat = self.feat(img)
        middle_feat = self.common[:1 if self.backbone_type == 'resnet50' else 7](low_feat)
        high_feat = self.common[1 if self.backbone_type == 'resnet50' else 7:](middle_feat)

        atte_low_feat = self.low_atte(low_feat, edge)
        atte_middle_feat = self.middle_atte(middle_feat, edge)
        atte_high_feat = self.high_atte(high_feat, edge)

        low_feat = torch.flatten(F.adaptive_max_pool2d(atte_low_feat, (1, 1)), start_dim=1)
        middle_feat = torch.flatten(F.adaptive_max_pool2d(atte_middle_feat, (1, 1)), start_dim=1)
        high_feat = torch.flatten(F.adaptive_max_pool2d(atte_high_feat, (1, 1)), start_dim=1)
        feat = torch.cat((low_feat, middle_feat, high_feat), dim=-1)
        proj = self.proj(feat)
        return F.normalize(proj, dim=-1)
