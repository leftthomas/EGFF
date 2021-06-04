import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, vgg16


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
        in_channel = 2048 if backbone_type == 'resnet50' else 512
        self.g = nn.Sequential(nn.Linear(in_channel, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, proj_dim, bias=True))

    def forward(self, x):
        feature = torch.flatten(F.adaptive_avg_pool2d(self.f(x), output_size=(1, 1)), start_dim=1)
        proj = self.g(feature)
        return F.normalize(proj, dim=-1)
