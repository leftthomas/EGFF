import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, vgg16


class GateAttention(nn.Module):
    def __init__(self, low_features, high_features):
        super(GateAttention, self).__init__()
        self.low_conv = nn.Conv2d(low_features, low_features, kernel_size=1, bias=False)
        self.high_conv = nn.Conv2d(high_features, low_features, kernel_size=1, bias=False)
        self.gate = nn.Conv2d(low_features, 1, kernel_size=1, bias=True)

    def forward(self, low_features, high_features):
        low_features = self.low_conv(low_features)
        high_features = self.high_conv(high_features)
        high_features = F.interpolate(high_features, size=low_features.size()[-2:], mode='bilinear')
        atte = torch.sigmoid(self.gate(F.relu(low_features + high_features)))
        feat = atte * low_features
        return atte, feat


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
            self.sketch_att = backbone[:5]
            self.photo_att = copy.deepcopy(backbone[:5])
            self.common = backbone[5:]
        elif backbone_type == 'vgg16':
            backbone = vgg16(pretrained=True).features
            self.sketch_att = backbone[:16]
            self.photo_att = copy.deepcopy(backbone[:16])
            self.common = backbone[16:-1]
        else:
            raise NotImplementedError('Not support {} as backbone'.format(backbone_type))
        # atte and proj
        self.sketch_atte = GateAttention(256, 2048 if backbone_type == 'resnet50' else 512)
        self.photo_atte = GateAttention(256, 2048 if backbone_type == 'resnet50' else 512)
        self.proj = nn.Linear(256 + 2048 if backbone_type == 'resnet50' else 256 + 512, proj_dim)

    def forward(self, img, domain):
        if torch.any(domain.bool()):
            sketch_feat = self.sketch_att(img[domain.bool()])
            sketch_g_feat = self.common(sketch_feat)
            sketch_atte_map, sketch_att_feat = self.sketch_atte(sketch_feat, sketch_g_feat)
        if torch.any(~domain.bool()):
            photo_feat = self.photo_att(img[~domain.bool()])
            photo_g_feat = self.common(photo_feat)
            photo_atte_map, photo_att_feat = self.photo_atte(photo_feat, photo_g_feat)

        if not torch.any(domain.bool()):
            g_feat = photo_g_feat
            att_feat = photo_att_feat
        if not torch.any(~domain.bool()):
            g_feat = sketch_g_feat
            att_feat = sketch_att_feat
        if torch.any(domain.bool()) and torch.any(~domain.bool()):
            g_feat = torch.cat((sketch_g_feat, photo_g_feat), dim=0)
            att_feat = torch.cat((sketch_att_feat, photo_att_feat), dim=0)

        feat = torch.flatten(F.adaptive_max_pool2d(g_feat, (1, 1)), start_dim=1)
        atte = torch.flatten(F.adaptive_max_pool2d(att_feat, (1, 1)), start_dim=1)
        proj = self.proj(torch.cat((feat, atte), dim=1))
        return F.normalize(proj, dim=-1)
