import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import torch
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from torchvision import transforms
import math
import numpy as np
from PIL import Image
import random

mean, std = {}, {}
mean['fungi']=[0.485, 0.456, 0.406]
std['fungi']=[0.229, 0.224, 0.225]

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

# def make_dataset_from_list(path, flist):
#     filelist_path = os.path.join(path, 'CUB_200_2011/filelist', flist)
#     path = os.path.join(path, 'CUB_200_2011/images')
#     with open(filelist_path) as f:
#         lines = f.readlines()
#         # imgs = [line.split(' ')[0] for line in lines]
#         imgs = [os.path.join(path, line.split(' ')[0]) for line in lines]
#         targets = [int(line.split(' ')[1].strip()) for line in lines]
#         imgs = np.array(imgs)
#         targets = np.array(targets)
#     return imgs, targets

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
def make_dataset_from_list(path, flist):
    filelist_path = os.path.join(path, 'semi_fungi/filelist', flist)
    path = os.path.join(path, 'semi_fungi')
    with open(filelist_path) as f:
        lines = f.readlines()
        # imgs = [line.split(' ')[0] for line in lines]
        imgs = [os.path.join(path, line.split(' ')[0]) for line in lines]
        targets = [int(line.split(' ')[-1].strip()) for line in lines]
        imgs = np.array(imgs)
        targets = np.array(targets)
    return imgs, targets

class SEMIFUNGIDatetset(BasicDataset):
    def __init__(self, root, transform, is_ulb, is_closed, args, alg, strong_transform=None, flist=None, train=True):
        super(SEMIFUNGIDatetset, self).__init__(alg=alg, data=None, is_ulb=is_ulb,
                                                transform=transform, strong_transform=strong_transform)
        self.root = root
        self.train = train
        if flist is not None:
            print('##### loading from flist #####')
            imgs, targets = make_dataset_from_list(self.root, flist)
        # else:
        #     if self.train:
        #         if is_ulb:
        #             _, _, imgs, targets = make_train_set(args, self.root)
        #         else:
        #             imgs, targets, _, _ = make_train_set(args, self.root)
        #     else:
        #         if is_closed:
        #             imgs, targets, _, _ = make_test_set(args, self.root)
        #         else:
        #             _, _, imgs, targets = make_test_set(args, self.root)
        if len(imgs) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)
        self.loader = pil_loader
        self.extensions = IMG_EXTENSIONS
        self.data = imgs
        self.targets = targets
        self.strong_transform = strong_transform
        def __sample__(self, idx):
            path, target = self.data[idx], self.targets[idx]
            img = self.loader(path)
            return img, target

def get_semifungi200(args, alg):
    img_size = args.img_size
    crop_ratio = args.crop_ratio
    transform_weak = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['fungi'], std['fungi'])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['fungi'], std['fungi'])
    ])
    # transform_weak = RandomCrop(img_size, crop_ratio)
    # transform_strong = AugRandomCrop(img_size, crop_ratio)
    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean['fungi'], std['fungi'])
    ])

    path = os.path.join(args.data_dir)

    lb_dset = SEMIFUNGIDatetset(root=path, 
                             flist='l_train_val.txt',
                             train=True,
                             transform=transform_weak, 
                             is_ulb=False, 
                             is_closed=False, 
                             args=args,
                             alg=alg
                             )
    ulb_dset = SEMIFUNGIDatetset(root=path, 
                              flist='u_train.txt',
                             train=True,
                             transform=transform_weak, 
                             strong_transform=transform_strong,
                             is_ulb=True, 
                             is_closed=False, 
                             args=args,
                             alg=alg
                             )
    eval_dset = SEMIFUNGIDatetset(root=path, 
                            flist='test.txt',
                            train=False,
                            transform=transform_val, 
                            is_ulb=False, 
                            is_closed=True, 
                            args=args,
                            alg=alg
                            )
    test_dset = SEMIFUNGIDatetset(root=path,
                            flist='test.txt', 
                            train=False,
                            transform=transform_val, 
                            is_ulb=False, 
                            is_closed=False, 
                            args=args,
                            alg=alg
                            )

    return lb_dset, ulb_dset, eval_dset, test_dset

def find_fungi_classnames(args):
    path = os.path.join(args.data_dir)
    classnames_path = os.path.join(path, 'semi_fungi/filelist/fungi_species_names.txt')
    with open(classnames_path) as f:
        lines = f.readlines()
        classnames = [line.strip('\n') for line in lines]
       
    return classnames