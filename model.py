import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, vgg16


class CBAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CBAM, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False))

    def forward(self, x):
        x_avg = self.fc(F.adaptive_avg_pool2d(x, (1, 1)))
        x_max = self.fc(F.adaptive_max_pool2d(x, (1, 1)))
        y = torch.sigmoid(x_avg + x_max)
        return x * y


class GateAttention(nn.Module):
    def __init__(self, low_features, high_features):
        super(GateAttention, self).__init__()
        self.low_conv = nn.Conv2d(low_features, low_features, kernel_size=1, bias=False)
        self.high_conv = nn.Conv2d(high_features, low_features, kernel_size=1, bias=False)
        self.channel_gate = CBAM(high_features)
        self.feature_gate = nn.Conv2d(low_features, 1, kernel_size=1, bias=True)

    def forward(self, low_features, high_features):
        low_features = self.low_conv(low_features)
        high_features = self.high_conv(self.channel_gate(high_features))
        high_features = F.interpolate(high_features, size=low_features.size()[-2:], mode='bilinear',
                                      align_corners=False)
        atte = torch.sigmoid(self.feature_gate(torch.relu(low_features + high_features)))
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
            self.feat = backbone[:5]
            self.common = backbone[5:]
        elif backbone_type == 'vgg16':
            backbone = vgg16(pretrained=True).features
            self.feat = backbone[:16]
            self.common = backbone[16:-1]
        else:
            raise NotImplementedError('Not support {} as backbone'.format(backbone_type))
        # atte and proj
        self.low_atte = GateAttention(256, 2048 if backbone_type == 'resnet50' else 512)
        self.middle_atte = GateAttention(512, 2048 if backbone_type == 'resnet50' else 512)
        self.proj = nn.Linear(256 + 512 + 2048 if backbone_type == 'resnet50' else 256 + 512 + 512, proj_dim)
        self.backbone_type = backbone_type

    def forward(self, img):
        low_feat = self.feat(img)
        middle_feat = self.common[:1 if self.backbone_type == 'resnet50' else 7](low_feat)
        high_feat = self.common[1 if self.backbone_type == 'resnet50' else 7:](middle_feat)

        att_low_map, att_low_feat = self.low_atte(low_feat, high_feat)
        att_middle_map, att_middle_feat = self.middle_atte(middle_feat, high_feat)

        low_feat = torch.flatten(F.adaptive_max_pool2d(att_low_feat, (1, 1)), start_dim=1)
        middle_feat = torch.flatten(F.adaptive_max_pool2d(att_middle_feat, (1, 1)), start_dim=1)
        high_feat = torch.flatten(F.adaptive_max_pool2d(high_feat, (1, 1)), start_dim=1)
        proj = self.proj(torch.cat((low_feat, middle_feat, high_feat), dim=1))
        return F.normalize(proj, dim=-1)
