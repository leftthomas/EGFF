import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, backbone_type, proj_dim):
        super(Model, self).__init__()
        # encoder
        self.backbone = timm.create_model('seresnet50' if backbone_type == 'resnet50' else 'vgg16_bn',
                                          features_only=True, out_indices=(2, 3, 4), pretrained=True)
        # atte and proj
        self.proj = nn.Linear(2048 if backbone_type == 'resnet50' else 512, proj_dim)
        self.backbone_type = backbone_type

    def forward(self, img):
        low_feat, middle_feat, high_feat = self.backbone(img)

        feat = torch.flatten(F.adaptive_max_pool2d(high_feat, (1, 1)), start_dim=1)
        proj = self.proj(feat)
        return F.normalize(proj, dim=-1)
