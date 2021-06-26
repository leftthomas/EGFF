import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import sobel
from torchvision.models import resnet50, vgg16


class Model(nn.Module):
    def __init__(self, backbone_type, edge_mode, proj_dim):
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
        self.g = nn.Linear(2048 if backbone_type == 'resnet50' else 512, proj_dim)
        self.backbone_type = backbone_type
        self.edge_mode = edge_mode

    def forward(self, x):
        if self.edge_mode != 'none':
            x = sobel(x)
        feature = torch.flatten(F.adaptive_avg_pool2d(self.f(x), output_size=(1, 1)), start_dim=1)
        proj = self.g(feature)
        return F.normalize(proj, dim=-1)
