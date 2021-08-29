import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimAM(nn.Module):
    def __init__(self, eps=1e-4):
        super(SimAM, self).__init__()
        self.eps = eps

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = d / (4 * (d.sum(dim=[2, 3], keepdim=True) / n + self.eps)) + 0.5
        return torch.sigmoid(y)


class EnergyAttention(nn.Module):
    def __init__(self, low_dim, high_dim):
        super(EnergyAttention, self).__init__()
        self.conv = nn.Conv2d(high_dim, low_dim, kernel_size=3, padding=1)

    def forward(self, low_feat, high_feat):
        high_feat = self.conv(high_feat)
        atte = F.interpolate(high_feat, low_feat.size()[-2:], mode='bilinear')
        low_feat = atte * low_feat
        return atte, low_feat


class Model(nn.Module):
    def __init__(self, backbone_type, proj_dim):
        super(Model, self).__init__()

        # backbone
        self.backbone = timm.create_model('seresnet50' if backbone_type == 'resnet50' else 'vgg16_bn',
                                          features_only=True, out_indices=(1, 2, 3, 4), pretrained=True)
        dims = [256, 512, 1024, 2048] if backbone_type == 'resnet50' else [128, 256, 512, 512]

        # atte
        self.energy_1 = EnergyAttention(dims[0], dims[1])
        self.energy_2 = EnergyAttention(dims[1], dims[2])
        self.energy_3 = EnergyAttention(dims[2], dims[3])

        # proj
        self.proj = nn.Linear(sum(dims), proj_dim)

    def forward(self, img):
        block_1_feat, block_2_feat, block_3_feat, block_4_feat = self.backbone(img)
        block_1_atte, block_1_feat = self.energy_1(block_1_feat, block_2_feat)
        block_2_atte, block_2_feat = self.energy_2(block_2_feat, block_3_feat)
        block_3_atte, block_3_feat = self.energy_3(block_3_feat, block_4_feat)

        feat = torch.cat((block_1_feat, block_2_feat, block_3_feat, block_4_feat), dim=-1)
        feat = torch.flatten(F.adaptive_max_pool2d(feat, (1, 1)), start_dim=1)
        proj = self.proj(feat)
        return F.normalize(proj, dim=-1)
