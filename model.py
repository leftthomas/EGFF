import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import sobel
from torchvision.models import resnet50, vgg16


class GlobalContext(nn.Module):
    def __init__(self):
        super(GlobalContext, self).__init__()
        self.conv_attn = nn.Conv2d(3, 1, 1, bias=True)

    def forward(self, x, edge):
        edge = F.interpolate(edge, size=x.size()[-2:], mode='bilinear', align_corners=False)
        attn = torch.sigmoid(self.conv_attn(edge))
        context = attn * x
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
        self.low_atte = GlobalContext()
        self.middle_atte = GlobalContext()
        self.high_atte = GlobalContext()
        self.proj = nn.Linear(2048 if backbone_type == 'resnet50' else 512, proj_dim)
        self.backbone_type = backbone_type

    def forward(self, img):
        edge = sobel(img)
        low_feat = self.feat(img)
        low_feat = self.low_atte(low_feat, edge)
        middle_feat = self.common[:1 if self.backbone_type == 'resnet50' else 7](low_feat)
        middle_feat = self.middle_atte(middle_feat, edge)
        high_feat = self.common[1 if self.backbone_type == 'resnet50' else 7:](middle_feat)
        high_feat = self.high_atte(high_feat, edge)

        feat = torch.flatten(F.adaptive_max_pool2d(high_feat, (1, 1)), start_dim=1)
        proj = self.proj(feat)
        return F.normalize(proj, dim=-1)
