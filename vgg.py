import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import pooling
from torchvision import models


class Embedding(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=None, normalized=True):
        super(Embedding, self).__init__()
        self.bn = nn.BatchNorm2d(in_dim, eps=1e-5)
        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.dropout = dropout
        self.normalized = normalized

    def forward(self, x):
        if self.dropout is not None:
            x = nn.Dropout(p=self.dropout)(x, inplace=True)
        x = self.linear(x)
        if self.normalized:
            norm = x.norm(dim=1, p=2, keepdim=True)
            x = x.div(norm.expand_as(x))
        return x


class SEblock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEblock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AttentionBlock(nn.Module):
    def __init__(self, in_features_l, in_features_g, attn_features, up_factor, normalize_attn=False, use_se=True):
        super(AttentionBlock, self).__init__()
        self.up_factor = up_factor
        self.use_se = use_se
        self.normalize_attn = normalize_attn
        self.W_l = nn.Conv2d(in_channels=in_features_l, out_channels=attn_features, kernel_size=1, padding=0,
                             bias=False)
        self.W_g = nn.Conv2d(in_channels=in_features_g, out_channels=attn_features, kernel_size=1, padding=0,
                             bias=False)
        self.phi = nn.Conv2d(in_channels=attn_features * 2, out_channels=1, kernel_size=1, padding=0, bias=True)
        if use_se:
            self.channel_gate = SEblock(in_features_l)

    def forward(self, l, g):
        N, C, W, H = l.size()
        if self.use_se:
            l = self.channel_gate(l)
        l_ = self.W_l(l)

        g_ = self.W_g(g)
        g_ = F.interpolate(g_, scale_factor=self.up_factor, mode='bilinear', align_corners=False)

        c = self.phi(F.relu(torch.cat([l_, g_], dim=1)))
        a = F.sigmoid(c)
        f = torch.mul(a.expand_as(l), l)

        output = F.adaptive_max_pool2d(f, (1, 1)).view(N, C)
        return a.view(N, 1, W, H), output


def vgg16(i=3):
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, dilation=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG16_V5_NET(nn.Module):
    def __init__(self, dim=512):
        super(VGG16_V5_NET, self).__init__()
        self.dim = dim
        self.vgg = vgg16()
        self.pool = pooling.GeM()
        self.atten1 = AttentionBlock(256, 512, 256, 4)
        self.atten2 = AttentionBlock(512, 512, 512, 2)
        self.classifier = Embedding(256 + 512 + 512, self.dim, normalized=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                try:
                    nn.init.constant_(m.bias, 0)
                except:
                    pass

    def forward(self, input):
        x1 = self.vgg[:16](input)
        x2 = self.vgg[16:23](x1)
        x3 = self.vgg[23:30](x2)

        g_fec = self.pool(x3).squeeze()
        c1, g1 = self.atten1(x1, x3)
        c2, g2 = self.atten2(x2, x3)
        feat = torch.cat([g_fec, g1, g2], dim=1)
        out = self.classifier(feat)
        return [out, c1, c2]  # c1 and c2 are heatmaps


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def VGG16_V5(dim=512, pretrain=True):
    model = VGG16_V5_NET(dim)
    if (pretrain):
        vgg_bn_pretrained = models.vgg16(pretrained=True)
        model.vgg.load_state_dict(vgg_bn_pretrained.features.state_dict())
    return model


if __name__ == '__main__':
    att = VGG16_V5_NET()
    print(att)
