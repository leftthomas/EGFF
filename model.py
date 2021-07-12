import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, vgg16


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
        # proj
        self.proj = nn.Linear(2048 if backbone_type == 'resnet50' else 512, proj_dim)

        self.backbone_type = backbone_type

    def forward(self, img, domain):
        if torch.any(domain.bool()):
            sketch_feat = self.sketch_att(img[domain.bool()])
        if torch.any(~domain.bool()):
            photo_feat = self.photo_att(img[~domain.bool()])

        if not torch.any(domain.bool()):
            x = photo_feat
        if not torch.any(~domain.bool()):
            x = sketch_feat
        if torch.any(domain.bool()) and torch.any(~domain.bool()):
            x = torch.cat((sketch_feat, photo_feat), dim=0)

        x = self.common(x)
        feat = torch.flatten(F.adaptive_max_pool2d(x, (1, 1)), start_dim=1)
        proj = self.proj(feat)
        return F.normalize(proj, dim=-1)
