import os
import bisect

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, USPS, SVHN, VisionDataset, utils, ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, Lambda
import h5py

class TruncatedDataset(Dataset):
    def __init__(self, dataset, dataset_name, indices=None):
        super().__init__()
        self.data = dataset.data
        if dataset_name == 'mnist':
            self.data = np.expand_dims(self.data, axis=3)
        if dataset_name in ['pascalvoc', 'imagenet']:
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
        self.labels = np.array(h5_file['labels'])

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        img = torch.from_numpy(self.data[index]).permute(2, 0, 1)
        target = self.labels[index]
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
        img = torch.from_numpy(self.data[index])
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
    def __init__(self, data_dir, img_tr=None, mask_tr=None, mode='trn'):
        super().__init__()
        self.img_tr = img_tr
        self.mask_tr = mask_tr

        h5_file = h5py.File(os.path.join(data_dir, 'CelebAMask-HQ/celeba2.hdf5'), 'r')
        if mode == 'trn':
            start_ndx = 0
            end_ndx = 24000
        else:
            start_ndx = 24000
            end_ndx = 30000
        self.indicies = list(range(start_ndx, end_ndx))
        self.data = h5_file['img']
        self.targets = h5_file['mask_one_hot']
        self.img_ids = h5_file['img_id']
        self.labels = h5_file['present_classes'][start_ndx:end_ndx]

    def __len__(self):
        return len(self.indicies)

    def __getitem__(self, index):
        index = self.indicies[index]
        img = self.data[index]
        mask = self.targets[index] > 0

        if self.img_tr is not None:
            img = self.img_tr(img)
        if self.mask_tr is not None:
            mask = self.mask_tr(mask)
        return img, mask
    
class MiniCOCODatase(Dataset):
    def __init__(self, data_dir, img_tr=None, mask_tr=None, mode='trn'):
        super().__init__()
        self.img_tr = img_tr
        self.mask_tr = mask_tr

        h5_file = h5py.File(os.path.join(data_dir, 'cocominitrain3.hdf5'), 'r')
        self.data = h5_file[mode]['img']
        self.targets = h5_file[mode]['mask']
        self.labels = h5_file[mode]['present_classes']

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img = self.data[index]
        mask = self.targets[index]

        if self.img_tr is not None:
            img = self.img_tr(img.transpose(1, 2, 0))
        if self.mask_tr is not None:
            mask = self.mask_tr(mask)
        return img, mask
    
class MiniCOCODatase2(Dataset):
    def __init__(self, data_dir, img_tr=None, mask_tr=None, mode='trn'):
        super().__init__()
        self.img_tr = img_tr
        self.mask_tr = mask_tr

        img_file = h5py.File(os.path.join(data_dir, 'cocominitrain3.hdf5'), 'r')
        # mask_file = h5py.File(os.path.join(data_dir, 'cocominitrain4.hdf5'), 'r')
        mask_file = h5py.File(os.path.join(data_dir, 'cocominitrain5.hdf5'), 'r')
        self.data = img_file[mode]['img']
        self.targets = mask_file[mode]['mask']
        self.labels = mask_file[mode]['present_classes']

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img = self.data[index]
        mask = self.targets[index]

        if self.img_tr is not None:
            img = self.img_tr(img.transpose(1, 2, 0))
        if self.mask_tr is not None:
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
    train_mean = [0.4914, 0.4822, 0.4465]
    train_std = [0.2470, 0.2435, 0.2616]
    train_transform = Compose([ToTensor(), Normalize(train_mean, train_std)])
    val_mean = [0.4914, 0.4822, 0.4465]
    val_std = [0.2470, 0.2435, 0.2616]
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
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        transforms = Compose([
            Resize((32, 32)),
            ToTensor(),
            Lambda(lambda x: x.repeat(3, 1, 1)),
            Normalize(mean, std)
        ])

        dataset = MNIST(root=data_dir, train=True, download=True, transform=transforms)
        dataset.targets = np.array(dataset.targets)
        val_dataset = MNIST(root=data_dir, train=False, transform=transforms)
        val_dataset.targets = np.array(val_dataset.targets)

    return dataset, val_dataset

def get_usps_dataset(data_dir):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transforms = Compose([Resize((32, 32)),
                          ToTensor(),
                          Lambda(lambda x: x.repeat(3, 1, 1)),
                          Normalize(mean, std)])

    dataset = USPS(root=data_dir, train=True, download=True, transform=transforms)
    dataset.targets = np.array(dataset.targets)
    val_dataset = USPS(root=data_dir, train=False, download=True, transform=transforms)
    val_dataset.targets = np.array(val_dataset.targets)

    return dataset, val_dataset

def get_svhn_dataset(data_dir):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transforms = Compose([Resize((32, 32)),
                          ToTensor(),
                          Normalize(mean, std)])

    dataset = SVHN(root=data_dir, split='train', download=True, transform=transforms)
    dataset.targets = np.array(dataset.labels)
    val_dataset = SVHN(root=data_dir, split='test', download=True, transform=transforms)
    val_dataset.targets = np.array(val_dataset.labels)

    return dataset, val_dataset

def get_syn_dataset(data_dir):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transforms = Compose([Resize((32, 32)),
                          ToTensor(),
                          Normalize(mean, std)])

    dataset = ImageFolder(root=os.path.join(data_dir, 'synthetic_digits', 'imgs_train'), transform=transforms)
    dataset.targets = np.array(dataset.targets)
    val_dataset = ImageFolder(root=os.path.join(data_dir, 'synthetic_digits', 'imgs_valid'), transform=transforms)
    val_dataset.targets = np.array(val_dataset.targets)

    return dataset, val_dataset

def get_digits_dataset(data_dir):
    mnist, val_mnist = get_mnist_datasets(data_dir)
    usps, val_usps = get_usps_dataset(data_dir)
    svhn, val_svhn = get_svhn_dataset(data_dir)
    syn, val_syn = get_syn_dataset(data_dir)

    return ConcatWithTargets([mnist, usps, svhn, syn]), ConcatWithTargets([val_mnist, val_usps, val_svhn, val_syn])


class ConcatWithTargets(Dataset):
    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets
        self.targets = np.concatenate([dset.targets for dset in datasets])
        self.cumulative_sizes = np.cumsum(np.array([len(dset) for dset in self.datasets]))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

def get_image_net_dataset(data_dir):
    dataset = ImageNetDataSet(data_dir=data_dir, mode='trn')
    val_dataset = ImageNetDataSet(data_dir=data_dir, mode='val')
    return dataset, val_dataset

def get_celeba_dataset(data_dir, img_tr=None, mask_tr=None):
    dataset = CelebAMask_HQDataset(data_dir, img_tr, mask_tr, 'trn')
    val_dataset = CelebAMask_HQDataset(data_dir, img_tr, mask_tr, 'val')
    return dataset, val_dataset

def get_minicoco_dataset(data_dir):
    mean = torch.tensor([0.40789654, 0.44719302, 0.47026115])
    std  = torch.tensor([0.28863828, 0.27408164, 0.27809835])
    transform = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    dataset = MiniCOCODatase2(data_dir, transform, None, 'trn')
    val_dataset = MiniCOCODatase2(data_dir, transform, None, 'val')
    return dataset, val_dataset