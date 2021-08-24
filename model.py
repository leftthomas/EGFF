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
    def __init__(self, base_dim, reduction=16):
        super(EnergyAttention, self).__init__()
        self.conv = nn.Conv2d(base_dim, base_dim // reduction, kernel_size=1)
        self.rev = nn.Conv2d(base_dim // reduction + 1, base_dim, kernel_size=1)
        self.gate = SimAM()

    def forward(self, feat, domain):
        tra = self.conv(feat)
        domain = domain.view(-1, 1, 1, 1).float().expand(-1, 1, *tra.size()[-2:])
        tra = torch.cat((tra, domain), dim=1)
        rev = self.rev(tra)
        atte = self.gate(rev)
        feat = atte * feat
        return atte, feat


class Model(nn.Module):
    def __init__(self, backbone_type, proj_dim):
        super(Model, self).__init__()
        # backbone
        backbone = list(timm.create_model('resnet50' if backbone_type == 'resnet50' else 'vgg16_bn',
                                          features_only=True, pretrained=True).children())
        if backbone_type == 'resnet50':
            indexes, dims = [3, 5, 6, 7, 8], [64, 256, 512, 1024, 2048]
        else:
            indexes, dims = [6, 13, 23, 33, 43], [64, 128, 256, 512, 512]

        # feat
        self.block_1 = nn.Sequential(*backbone[:indexes[0]])
        self.block_2 = nn.Sequential(*backbone[indexes[0]:indexes[1]])
        self.block_3 = nn.Sequential(*backbone[indexes[1]:indexes[2]])
        self.block_4 = nn.Sequential(*backbone[indexes[2]:indexes[3]])
        self.block_5 = nn.Sequential(*backbone[indexes[3]:indexes[4]])

        # atte
        self.energy_1 = EnergyAttention(dims[0])
        self.energy_2 = EnergyAttention(dims[1])
        self.energy_3 = EnergyAttention(dims[2])
        self.energy_4 = EnergyAttention(dims[3])
        self.energy_5 = EnergyAttention(dims[4])

        # proj
        self.proj = nn.Linear(2048 if backbone_type == 'resnet50' else 512, proj_dim)

    def forward(self, img, domain):
        block_1_feat = self.block_1(img)
        block_1_atte, block_1_feat = self.energy_1(block_1_feat, domain)
        block_2_feat = self.block_2(block_1_feat)
        block_2_atte, block_2_feat = self.energy_2(block_2_feat, domain)
        block_3_feat = self.block_3(block_2_feat)
        block_3_atte, block_3_feat = self.energy_3(block_3_feat, domain)
        block_4_feat = self.block_4(block_3_feat)
        block_4_atte, block_4_feat = self.energy_4(block_4_feat, domain)
        block_5_feat = self.block_5(block_4_feat)
        block_5_atte, block_5_feat = self.energy_5(block_5_feat, domain)

        feat = torch.flatten(F.adaptive_max_pool2d(block_5_feat, (1, 1)), start_dim=1)
        proj = self.proj(feat)
        return F.normalize(proj, dim=-1)
