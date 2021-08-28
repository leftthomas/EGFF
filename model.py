import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimAM(nn.Module):
    def __init__(self, eps=1e-4):
        super(SimAM, self).__init__()
        self.eps = eps

    def forward(self, feat):
        b, c, h, w = feat.size()
        n = h * w
        d = (feat - feat.mean(dim=[2, 3], keepdim=True)).pow(2)
        atte = d / (4 * (d.sum(dim=[2, 3], keepdim=True) / n + self.eps)) + 0.5
        atte = torch.sigmoid(atte)
        feat = atte * feat
        return atte, feat


class Model(nn.Module):
    def __init__(self, backbone_type, proj_dim):
        super(Model, self).__init__()
        # backbone
        backbone = list(timm.create_model('resnet50' if backbone_type == 'resnet50' else 'vgg16_bn',
                                          features_only=True, pretrained=True).children())
        indexes = [6, 7, 8] if backbone_type == 'resnet50' else [23, 33, 43]

        # feat
        self.block_1 = nn.Sequential(*backbone[:indexes[0]])
        self.block_2 = nn.Sequential(*backbone[indexes[0]:indexes[1]])
        self.block_3 = nn.Sequential(*backbone[indexes[1]:indexes[2]])

        # atte
        self.energy_1 = SimAM()
        self.energy_2 = SimAM()
        self.energy_3 = SimAM()

        # proj
        self.proj = nn.Linear(2048 if backbone_type == 'resnet50' else 512, proj_dim)

    def forward(self, img):
        block_1_feat = self.block_1(img)
        block_1_atte, block_1_feat = self.energy_1(block_1_feat)
        block_2_feat = self.block_2(block_1_feat)
        block_2_atte, block_2_feat = self.energy_2(block_2_feat)
        block_3_feat = self.block_3(block_2_feat)
        block_3_atte, block_3_feat = self.energy_3(block_3_feat)

        feat = torch.flatten(F.adaptive_max_pool2d(block_3_feat, (1, 1)), start_dim=1)
        proj = self.proj(feat)
        return F.normalize(proj, dim=-1)
