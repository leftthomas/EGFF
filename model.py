import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, vgg16


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AttentionBlock(nn.Module):
    def __init__(self, in_features_l, in_features_g, attn_features, up_factor):
        super(AttentionBlock, self).__init__()
        self.up_factor = up_factor
        self.W_l = nn.Conv2d(in_channels=in_features_l, out_channels=attn_features, kernel_size=1, padding=0,
                             bias=False)
        self.W_g = nn.Conv2d(in_channels=in_features_g, out_channels=attn_features, kernel_size=1, padding=0,
                             bias=False)
        self.phi = nn.Conv2d(in_channels=attn_features, out_channels=1, kernel_size=1, padding=0, bias=True)
        self.channel_gate = SEBlock(in_features_l)

    def forward(self, l, g):
        B, C, H, W = l.size()

        l = self.channel_gate(l)
        l_ = self.W_l(l)

        g_ = self.W_g(g)
        g_ = F.interpolate(g_, scale_factor=self.up_factor, mode='bilinear', align_corners=False)

        c = self.phi(F.relu(l_ + g_))
        a = torch.sigmoid(c)
        f = torch.mul(a.expand_as(l), l)

        output = F.adaptive_max_pool2d(f, (1, 1)).view(B, C)
        return a.view(B, 1, H, W), output


class Model(nn.Module):
    def __init__(self, backbone_type, proj_dim):
        super(Model, self).__init__()
        # encoder
        if backbone_type == 'resnet50':
            self.f = []
            for module in resnet50(pretrained=True).children():
                if not isinstance(module, (nn.Linear, nn.AdaptiveAvgPool2d)):
                    self.f.append(module)
            self.f = nn.Sequential(*self.f)
        elif backbone_type == 'vgg16':
            self.f = vgg16(pretrained=True).features
        else:
            raise NotImplementedError('Not support {} as backbone'.format(backbone_type))
        # head
        self.g = nn.Linear(256 + 512 + 2048 if backbone_type == 'resnet50' else 256 + 512 + 512, proj_dim)
        self.backbone_type = backbone_type
        self.pool = nn.AdaptiveMaxPool2d(1)
        if self.backbone_type == 'resnet50':
            self.atten_1 = AttentionBlock(256, 2048, 256, 8)
            self.atten_2 = AttentionBlock(512, 2048, 512, 4)
        else:
            self.atten_1 = AttentionBlock(256, 512, 256, 4)
            self.atten_2 = AttentionBlock(512, 512, 512, 2)

    def forward(self, x):
        if self.backbone_type == 'resnet50':
            x1 = self.f[:5](x)
            x2 = self.f[5:6](x1)
            x3 = self.f[6:8](x2)
        else:
            x1 = self.f[:16](x)
            x2 = self.f[16:23](x1)
            x3 = self.f[23:30](x2)
        g_fec = self.pool(x3).squeeze()
        c1, g1 = self.atten_1(x1, x3)
        c2, g2 = self.atten_2(x2, x3)
        feat = torch.cat([g_fec, g1, g2], dim=1)
        proj = self.g(feat)
        return F.normalize(proj, dim=-1)


class SimCLRLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    def forward(self, proj_1, proj_2):
        batch_size = proj_1.size(0)
        # [2*B, Dim]
        out = torch.cat([proj_1, proj_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(proj_1 * proj_2, dim=-1) / self.temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        return loss
