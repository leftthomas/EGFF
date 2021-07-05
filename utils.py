import glob
import os
import random

from PIL import Image
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator, precision_at_k
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def get_transform(split='train'):
    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    else:
        return transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


class DomainDataset(Dataset):
    def __init__(self, data_root, data_name, split='train'):
        super(DomainDataset, self).__init__()

        dirname = 'photo' if split == 'train' else '*'
        self.images = sorted(glob.glob(os.path.join(data_root, data_name, split, dirname, '*', '*.jpg')))
        self.transform = get_transform(split)
        self.split = split

        self.labels, self.classes = [], {}
        i = 0
        for img in self.images:
            label = os.path.dirname(img).split('/')[-1]
            if label not in self.classes:
                self.classes[label] = i
                i += 1
            self.labels.append(self.classes[label])

    def __getitem__(self, index):
        img_name = self.images[index]
        label = self.labels[index]
        img = Image.open(img_name)
        img = self.transform(img)
        if self.split == 'train':
            sketches = sorted(glob.glob(os.path.join(os.path.dirname(img_name).replace('photo', 'sketch'), '*.jpg')))
            sketch_name = random.choice(sketches)
            sketch = Image.open(sketch_name)
            sketch = self.transform(sketch)
            return img, sketch, label
        else:
            domain = 0 if 'photo' in img_name else 1
            return img, domain, label

    def __len__(self):
        return len(self.images)


class MetricCalculator(AccuracyCalculator):
    def calculate_precision_at_100(self, knn_labels, query_labels, **kwargs):
        return precision_at_k(knn_labels, query_labels[:, None], 100, self.avg_of_avgs, self.label_comparison_fn)

    def calculate_precision_at_200(self, knn_labels, query_labels, **kwargs):
        return precision_at_k(knn_labels, query_labels[:, None], 200, self.avg_of_avgs, self.label_comparison_fn)

    def requires_knn(self):
        return super().requires_knn() + ["precision_at_100", "precision_at_200"]


def compute_metric(vectors, domains, labels):
    calculator_200 = MetricCalculator(include=['mean_average_precision', 'precision_at_100', 'precision_at_200'], k=200)
    calculator_all = MetricCalculator(include=['mean_average_precision'])
    acc = {}

    photo_vectors = vectors[domains == 0]
    sketch_vectors = vectors[domains == 1]
    photo_labels = labels[domains == 0]
    sketch_labels = labels[domains == 1]
    map_200 = calculator_200.get_accuracy(sketch_vectors, photo_vectors, sketch_labels, photo_labels, False)
    map_all = calculator_all.get_accuracy(sketch_vectors, photo_vectors, sketch_labels, photo_labels, False)

    acc['P@100'] = map_200['precision_at_100']
    acc['P@200'] = map_200['precision_at_200']
    acc['mAP@200'] = map_200['mean_average_precision']
    acc['mAP@all'] = map_all['mean_average_precision']
    # the mean value is chosen as the representative of precise
    acc['precise'] = (acc['P@100'] + acc['P@200'] + acc['mAP@200'] + acc['mAP@all']) / 4
    return acc
