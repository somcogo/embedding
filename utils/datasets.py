import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
import h5py

class TruncatedDataset(Dataset):
    def __init__(self, dataset, dataset_name, indices=None):
        super().__init__()
        self.data = dataset.data
        if dataset_name == 'pascalvoc':
            self.labels = dataset.labels
        else:
            self.labels = dataset.targets

        if indices is not None:
            self.indices = indices
            self.data = self.data[indices]
            self.labels = self.labels[indices]
        else:
            self.indices = range(len(dataset))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]
        return img, label

class ImageNetDataSet(Dataset):
    def __init__(self, data_dir, mode):
        super().__init__()
        h5_file = h5py.File(os.path.join(data_dir, 'tiny_imagenet_{}.hdf5'.format(mode)), 'r')
        self.data = np.array(h5_file['data'])
        self.targets = np.array(h5_file['labels'])

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        img = torch.from_numpy(self.data[index]).permute(2, 0, 1)
        target = self.targets[index]
        return img, target

def get_cifar10_datasets(data_dir):
    train_mean = [0.4914, 0.4822, 0.4465]
    train_std = [0.2470, 0.2435, 0.2616]
    train_transform = Compose([ToTensor(), Normalize(train_mean, train_std)])
    val_mean = [0.4942, 0.4851, 0.4504]
    val_std = [0.2467, 0.2429, 0.2616]
    val_transform = Compose([ToTensor(), Normalize(val_mean, val_std)])

    dataset = CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    dataset.targets = np.array(dataset.targets)
    val_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=val_transform)
    val_dataset.targets = np.array(val_dataset.targets)

    return dataset, val_dataset

def get_cifar100_datasets(data_dir):
    train_mean = [0.5071, 0.4866, 0.4409]
    train_std = [0.2673, 0.2564, 0.2762]
    train_transform = Compose([ToTensor(), Normalize(train_mean, train_std)])
    val_mean = [0.5088, 0.4874, 0.4419]
    val_std = [0.2683, 0.2574, 0.2771]
    val_transform = Compose([ToTensor(), Normalize(val_mean, val_std)])

    dataset = CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    dataset.targets = np.array(dataset.targets)
    val_dataset = CIFAR100(root=data_dir, train=False, download=True, transform=val_transform)
    val_dataset.targets = np.array(val_dataset.targets)

    return dataset, val_dataset

def get_mnist_datasets(data_dir):
    mean = 0.1307
    std = 0.3081
    transforms = Compose([ToTensor(), Normalize(mean, std)])

    dataset = MNIST(root=data_dir, train=True, download=True, transform=transforms)
    dataset.targets = np.array(dataset.targets)
    dataset.data = dataset.data.unsqueeze(dim=1).permute((0, 2, 3, 1))
    val_dataset = MNIST(root=data_dir, train=False, transform=transforms)
    val_dataset.targets = np.array(val_dataset.targets)
    val_dataset.data = val_dataset.data.unsqueeze(dim=1).permute((0, 2, 3 ,1))

    return dataset, val_dataset

def get_image_net_dataset(data_dir):
    dataset = ImageNetDataSet(data_dir=data_dir, mode='trn')
    val_dataset = ImageNetDataSet(data_dir=data_dir, mode='val')
    return dataset, val_dataset