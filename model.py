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
            self.sketch_feat = backbone[:5]
            self.photo_feat = copy.deepcopy(backbone[:5])
            self.common = backbone[5:]
        elif backbone_type == 'vgg16':
            backbone = vgg16(pretrained=True).features
            self.sketch_feat = backbone[:16]
            self.photo_feat = copy.deepcopy(backbone[:16])
            self.common = backbone[16:-1]
        else:
            raise NotImplementedError('Not support {} as backbone'.format(backbone_type))
        # atte and proj
        self.sketch_low_atte = GateAttention(256, 2048 if backbone_type == 'resnet50' else 512)
        self.photo_low_atte = GateAttention(256, 2048 if backbone_type == 'resnet50' else 512)
        self.sketch_middle_atte = GateAttention(512, 2048 if backbone_type == 'resnet50' else 512)
        self.photo_middle_atte = GateAttention(512, 2048 if backbone_type == 'resnet50' else 512)
        self.proj = nn.Linear(256 + 512 + 2048 if backbone_type == 'resnet50' else 256 + 512 + 512, proj_dim)
        self.backbone_type = backbone_type

    def forward(self, img, domain):
        if torch.any(domain.bool()):
            sketch_low_feat = self.sketch_feat(img[domain.bool()])
            sketch_middle_feat = self.common[:1 if self.backbone_type == 'resnet50' else 7](sketch_low_feat)
            sketch_high_feat = self.common[1 if self.backbone_type == 'resnet50' else 7:](sketch_middle_feat)
            sketch_att_low_map, sketch_att_low_feat = self.sketch_low_atte(sketch_low_feat, sketch_high_feat)
            sketch_att_middle_map, sketch_att_middle_feat = self.sketch_middle_atte(sketch_middle_feat,
                                                                                    sketch_high_feat)
        if torch.any(~domain.bool()):
            photo_low_feat = self.photo_feat(img[~domain.bool()])
            photo_middle_feat = self.common[:1 if self.backbone_type == 'resnet50' else 7](photo_low_feat)
            photo_high_feat = self.common[1 if self.backbone_type == 'resnet50' else 7:](photo_middle_feat)
            photo_att_low_map, photo_att_low_feat = self.photo_low_atte(photo_low_feat, photo_high_feat)
            photo_att_middle_map, photo_att_middle_feat = self.photo_middle_atte(photo_middle_feat, photo_high_feat)

        if not torch.any(domain.bool()):
            high_feat = photo_high_feat
            att_low_feat = photo_att_low_feat
            att_middle_feat = photo_att_middle_feat
        if not torch.any(~domain.bool()):
            high_feat = sketch_high_feat
            att_low_feat = sketch_att_low_feat
            att_middle_feat = sketch_att_middle_feat
        if torch.any(domain.bool()) and torch.any(~domain.bool()):
            high_feat = torch.cat((sketch_high_feat, photo_high_feat), dim=0)
            att_low_feat = torch.cat((sketch_att_low_feat, photo_att_low_feat), dim=0)
            att_middle_feat = torch.cat((sketch_att_middle_feat, photo_att_middle_feat), dim=0)

        feat = torch.flatten(F.adaptive_max_pool2d(high_feat, (1, 1)), start_dim=1)
        low_att = torch.flatten(F.adaptive_max_pool2d(att_low_feat, (1, 1)), start_dim=1)
        middle_att = torch.flatten(F.adaptive_max_pool2d(att_middle_feat, (1, 1)), start_dim=1)
        proj = self.proj(torch.cat((feat, low_att, middle_att), dim=1))
        return F.normalize(proj, dim=-1)
