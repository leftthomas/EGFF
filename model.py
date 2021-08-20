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


class EnergyFFM(nn.Module):
    def __init__(self, base_dim, atte_dim):
        super(EnergyFFM, self).__init__()
        self.base_conv = nn.Conv2d(base_dim, base_dim, kernel_size=1, bias=False)
        self.atte_conv = nn.Conv2d(atte_dim, base_dim, kernel_size=1, bias=False)
        self.gate = SimAM()

    def forward(self, base_feat, atte_feat):
        base_tra = self.base_conv(base_feat)
        atte_tra = self.atte_conv(atte_feat)
        atte_tra = F.interpolate(atte_tra, size=base_tra.size()[-2:], mode='bilinear')
        atte = self.gate(base_tra + atte_tra)
        feat = atte * base_feat
        return atte, feat


class Model(nn.Module):
    def __init__(self, backbone_type, proj_dim):
        super(Model, self).__init__()
        # encoder
        self.backbone = timm.create_model('seresnet50' if backbone_type == 'resnet50' else 'vgg16_bn',
                                          features_only=True, out_indices=(2, 3, 4), pretrained=True)
        # atte and proj
        if backbone_type == 'resnet50':
            low_dim, middle_dim, high_dim = 512, 1024, 2048
        else:
            low_dim, middle_dim, high_dim = 256, 512, 512
        self.low_atte = EnergyFFM(low_dim, high_dim)
        self.middle_atte = EnergyFFM(middle_dim, high_dim)
        self.high_atte = EnergyFFM(high_dim, high_dim)
        self.proj = nn.Linear(low_dim + middle_dim + high_dim, proj_dim)
        self.backbone_type = backbone_type

    def forward(self, img):
        low_feat, middle_feat, high_feat = self.backbone(img)
        low_atte, low_feat = self.low_atte(low_feat, high_feat)
        middle_atte, middle_feat = self.middle_atte(middle_feat, high_feat)
        high_atte, high_feat = self.high_atte(high_feat, high_feat)

        low_feat = torch.flatten(F.adaptive_max_pool2d(low_feat, (1, 1)), start_dim=1)
        middle_feat = torch.flatten(F.adaptive_max_pool2d(middle_feat, (1, 1)), start_dim=1)
        high_feat = torch.flatten(F.adaptive_max_pool2d(high_feat, (1, 1)), start_dim=1)
        feat = torch.cat((low_feat, middle_feat, high_feat), dim=-1)
        proj = self.proj(feat)
        return F.normalize(proj, dim=-1)
