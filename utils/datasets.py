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
        if dataset_name == 'mnist':
            self.data = np.expand_dims(self.data, axis=3)
        if dataset_name == 'pascalvoc':
            self.labels = dataset.labels
        else:
            self.labels = dataset.targets

        self.indices = indices if indices is not None else list(range(len(dataset)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        img_index = self.indices[index]
        img = self.data[img_index]
        label = self.labels[img_index]
        return img, label, img_index
    
class MergedDataset(Dataset):
    def __init__(self, dataset1, dataset2, dataset_name):
        super().__init__()
        if dataset_name == 'mnist':
            self.data = torch.cat([dataset1.data, dataset2.data], dim=0)
        else:
            self.data = np.concatenate([dataset1.data, dataset2.data], axis=0)
        self.targets = np.concatenate([dataset1.labels, dataset2.labels], axis=0)

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        return self.data[index], self.targets[index]

class ImageNetDataSet(Dataset):
    def __init__(self, data_dir, mode):
        super().__init__()
        h5_file = h5py.File(os.path.join(data_dir, 'tiny_imagenet_{}.hdf5'.format(mode)), 'r')
        self.data = h5_file['data']
        self.targets = np.array(h5_file['labels'])

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        img = torch.from_numpy(self.data[index]).permute(2, 0, 1)
        target = self.targets[index]
        return img, target

class CIFAR10DataSet(Dataset):
    def __init__(self, data_dir, mode):
        super().__init__()
        h5_file = h5py.File(os.path.join(data_dir, 'cifar10_{}.hdf5'.format(mode)), 'r')
        self.data = h5_file['data']
        self.targets = np.array(h5_file['labels'])

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        img = torch.from_numpy(np.transpose(self.data[index], (1, 2, 0)))
        target = self.targets[index]
        return img, target

class MNISTDataSet(Dataset):
    def __init__(self, data_dir, mode):
        super().__init__()
        h5_file = h5py.File(os.path.join(data_dir, 'mnist_{}.hdf5'.format(mode)), 'r')
        self.data = h5_file['data']
        self.targets = np.array(h5_file['targets'])

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        img = torch.from_numpy(self.data[index])
        target = self.targets[index]
        return img, target
    
class CelebAMask_HQDataset(Dataset):
    def __init__(self, data_dir, img_tr, mask_tr, mode):
        super().__init__()
        self.img_tr = img_tr
        self.mask_tr = mask_tr

        h5_file = h5py.File(os.path.join(data_dir, 'celeba.hdf5'), 'r')
        if mode == 'trn':
            start_ndx = 0
            end_ndx = 24000
        else:
            start_ndx = 24000
            end_ndx = 30000
        self.data = h5_file['img'][start_ndx:end_ndx]
        self.labels = h5_file['mask'][start_ndx:end_ndx]
        self.img_ids = h5_file['img_id'][start_ndx:end_ndx]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img = self.data[index]
        mask = self.labels[index]

        img = self.img_tr(img)
        mask = self.mask_tr(mask)
        return img, mask

def get_cifar10_datasets(data_dir, use_hdf5=False):
    if use_hdf5:
        dataset = CIFAR10DataSet(data_dir, 'trn')
        val_dataset = CIFAR10DataSet(data_dir, 'val')
    else:
        train_mean = [0.4914, 0.4822, 0.4465]
        train_std = [0.2470, 0.2435, 0.2616]
        train_transform = Compose([ToTensor(), Normalize(train_mean, train_std)])
        val_mean = [0.4942, 0.4851, 0.4504]
        val_std = [0.2467, 0.2429, 0.2616]
        val_transform = Compose([ToTensor(), Normalize(val_mean, val_std)])

        dataset = CIFAR10(root=data_dir, train=True, download=False, transform=train_transform)
        dataset.targets = np.array(dataset.targets)
        val_dataset = CIFAR10(root=data_dir, train=False, download=False, transform=val_transform)
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

def get_mnist_datasets(data_dir, use_hdf5=False):
    if use_hdf5:
        dataset = MNISTDataSet(data_dir, 'trn')
        val_dataset = MNISTDataSet(data_dir, 'val')
    else:
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

def get_celeba_dataset(data_dir, img_tr, mask_tr):
    dataset = CelebAMask_HQDataset(data_dir, img_tr, mask_tr, 'trn')
    val_dataset = CelebAMask_HQDataset(data_dir, img_tr, mask_tr, 'val')
    return dataset, val_dataset