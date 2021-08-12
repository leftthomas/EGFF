import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimAM(torch.nn.Module):
    def __init__(self, eps=1e-4):
        super(SimAM, self).__init__()
        self.eps = eps

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = d / (4 * (d.sum(dim=[2, 3], keepdim=True) / n + self.eps)) + 0.5
        return torch.flatten(F.adaptive_max_pool2d(x * torch.sigmoid(y), (1, 1)), start_dim=1)


class Model(nn.Module):
    def __init__(self, backbone_type, proj_dim):
        super(Model, self).__init__()
        # encoder
        self.backbone = timm.create_model('seresnet50' if backbone_type == 'resnet50' else 'vgg16_bn',
                                          features_only=True, out_indices=(2, 3, 4), pretrained=True)
        # atte and proj
        self.atten = SimAM()
        self.proj = nn.Linear(512 + 1024 + 2048 if backbone_type == 'resnet50' else 256 + 512 + 512, proj_dim)
        self.backbone_type = backbone_type

    def forward(self, img):
        low_feat, middle_feat, high_feat = self.backbone(img)
        low_feat, middle_feat, high_feat = self.atten(low_feat), self.atten(middle_feat), self.atten(high_feat)
        feat = torch.cat((low_feat, middle_feat, high_feat), dim=-1)
        proj = self.proj(feat)
        return F.normalize(proj, dim=-1)
