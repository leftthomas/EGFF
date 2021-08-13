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
        return y


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
        self.atte = SimAM()
        self.low_tra = nn.Conv2d(high_dim, low_dim, kernel_size=1, bias=False)
        self.middle_tra = nn.Conv2d(high_dim, middle_dim, kernel_size=1, bias=False)
        self.high_tra = nn.Conv2d(high_dim, high_dim, kernel_size=1, bias=False)
        self.proj = nn.Linear(low_dim + middle_dim + high_dim, proj_dim)
        self.backbone_type = backbone_type

    def forward(self, img):
        low_feat, middle_feat, high_feat = self.backbone(img)
        atte = self.atte(high_feat)

        low_atte = self.low_tra(atte)
        middle_atte = self.middle_tra(atte)
        high_atte = self.high_tra(atte)

        low_atte = torch.sigmoid(F.interpolate(low_atte, size=low_feat.size()[-2:]))
        middle_atte = torch.sigmoid(F.interpolate(middle_atte, size=middle_feat.size()[-2:]))
        high_atte = torch.sigmoid(F.interpolate(high_atte, size=high_feat.size()[-2:]))

        low_feat = low_feat * low_atte
        middle_feat = middle_feat * middle_atte
        high_feat = high_feat * high_atte

        low_feat = torch.flatten(F.adaptive_max_pool2d(low_feat, (1, 1)), start_dim=1)
        middle_feat = torch.flatten(F.adaptive_max_pool2d(middle_feat, (1, 1)), start_dim=1)
        high_feat = torch.flatten(F.adaptive_max_pool2d(high_feat, (1, 1)), start_dim=1)
        feat = torch.cat((low_feat, middle_feat, high_feat), dim=-1)

        proj = self.proj(feat)
        return F.normalize(proj, dim=-1)
