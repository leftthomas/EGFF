import argparse
import itertools
import os
import random

import numpy as np
import pandas as pd
import torch
from pytorch_metric_learning.losses import ProxyAnchorLoss
from torch.backends import cudnn
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model import Model
from utils import DomainDataset, compute_metric

# for reproducibility
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
cudnn.deterministic = True
cudnn.benchmark = False


# train for one epoch
def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader, dynamic_ncols=True)
    for img, domain, label, img_name in train_bar:
        proj = net(img.cuda())
        loss = loss_criterion(proj, label.cuda())
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += img.size(0)
        total_loss += loss.item() * img.size(0)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# val for one epoch
def val(net, data_loader):
    net.eval()
    vectors = []
    with torch.no_grad():
        for img, domain, label, img_name in tqdm(data_loader, desc='Feature extracting', dynamic_ncols=True):
            vectors.append(net(img.cuda()))
        vectors = torch.cat(vectors, dim=0)
        domains = data_loader.dataset.domains
        labels = data_loader.dataset.labels
        acc = compute_metric(vectors, domains, labels)
        results['P@100'].append(acc['P@100'] * 100)
        results['P@200'].append(acc['P@200'] * 100)
        results['mAP@200'].append(acc['mAP@200'] * 100)
        results['mAP@all'].append(acc['mAP@all'] * 100)
        print('Val Epoch: [{}/{}] | P@100:{:.1f}% | P@200:{:.1f}% | mAP@200:{:.1f}% | mAP@all:{:.1f}%'
              .format(epoch, epochs, acc['P@100'] * 100, acc['P@200'] * 100, acc['mAP@200'] * 100,
                      acc['mAP@all'] * 100))
    return acc['precise'], vectors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Model')
    # common args
    parser.add_argument('--data_root', default='data', type=str, help='Datasets root path')
    parser.add_argument('--data_name', default='sketchy', type=str, choices=['sketchy', 'tuberlin'],
                        help='Dataset name')
    parser.add_argument('--backbone_type', default='resnet50', type=str, choices=['resnet50', 'vgg16'],
                        help='Backbone type')
    parser.add_argument('--edge_mode', default='auto', type=str, choices=['auto', 'both', 'photo', 'none'],
                        help='Edge extraction mode')
    parser.add_argument('--proj_dim', default=512, type=int, help='Projected embedding dim')
    parser.add_argument('--batch_size', default=64, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs over the model to train')
    parser.add_argument('--save_root', default='result', type=str, help='Result saved root path')

    # args parse
    args = parser.parse_args()
    data_root, data_name, backbone_type, edge_mode = args.data_root, args.data_name, args.backbone_type, args.edge_mode
    proj_dim, batch_size, epochs, save_root = args.proj_dim, args.batch_size, args.epochs, args.save_root

    # data prepare
    train_data = DomainDataset(data_root, data_name, edge_mode, split='train')
    val_data = DomainDataset(data_root, data_name, edge_mode, split='val')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)

    # model and loss setup
    model = Model(backbone_type, proj_dim).cuda()
    loss_criterion = ProxyAnchorLoss(len(train_data.classes), proj_dim).cuda()
    # optimizer config
    optimizer = Adam(itertools.chain(model.parameters(), loss_criterion.parameters()), lr=1e-3, weight_decay=1e-6)

    # training loop
    results = {'train_loss': [], 'val_precise': [], 'P@100': [], 'P@200': [], 'mAP@200': [], 'mAP@all': []}
    save_name_pre = '{}_{}_{}_{}'.format(data_name, backbone_type, edge_mode, proj_dim)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    best_precise = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        val_precise, features = val(model, val_loader)
        results['val_precise'].append(val_precise * 100)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('{}/{}_results.csv'.format(save_root, save_name_pre), index_label='epoch')

        if val_precise > best_precise:
            best_precise = val_precise
            torch.save(model.state_dict(), '{}/{}_model.pth'.format(save_root, save_name_pre))
            torch.save(features, '{}/{}_vectors.pth'.format(save_root, save_name_pre))
