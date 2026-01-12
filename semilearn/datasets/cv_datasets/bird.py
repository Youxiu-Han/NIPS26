import copy
import os

import numpy as np
from PIL import Image
from torchvision import transforms
import math

from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from .datasetbase import BasicDataset
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets.folder import default_loader, make_dataset

mean, std = {}, {}
mean['bird']=[0.485, 0.456, 0.406]
std['bird']=[0.229, 0.224, 0.225]

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
def make_dataset(directory, class_to_idx):
    imgs = []
    targets = []
    directory = os.path.expanduser(directory)
    for target in os.listdir(directory):
        d = os.path.join(directory, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    imgs.append(path)
                    targets.append(class_to_idx[target])
    imgs = np.array(imgs)
    targets = np.array(targets)

    return imgs, targets

def make_dataset_from_list(flist):
    # print('flist', flist)
    with open(flist) as f:
        lines = f.readlines()
        imgs = [line.split(' ')[0] for line in lines]
        targets = [int(line.split(' ')[1].strip()) for line in lines]
        imgs = np.array(imgs)
        targets = np.array(targets)
    return imgs, targets

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class BIRD30Datetset(BasicDataset):
    def __init__(self, root, transform, is_ulb, alg, strong_transform=None, flist=None):
        super(BIRD30Datetset, self).__init__(alg=alg, data=None, is_ulb=is_ulb,
                                                transform=transform, strong_transform=strong_transform)
       
        self.root = root

        if flist is not None:
            imgs, targets = make_dataset_from_list(flist)
        else:
            print('please provide flist')
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

def get_bird200(args, alg, name, num_labels, num_classes, data_dir='./data'):
    img_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['bird'], std['bird'])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['bird'], std['bird'])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean['bird'], std['bird'])
    ])

    data_dir = os.path.join(data_dir, 'bird')
    
    lb_dset = BIRD30Datetset(root=os.path.join(data_dir, "images"), 
                             transform=transform_weak, 
                             is_ulb=False, 
                             alg=alg, 
                             flist=os.path.join(data_dir, f'filelist/train_labeled_{num_labels}.txt'))
                            # flist=os.path.join(data_dir, f'filelist/train_labeled_full.txt'))
    ulb_dset = BIRD30Datetset(root=os.path.join(data_dir, "images"), 
                             transform=transform_weak, 
                             is_ulb=True, 
                             alg=alg, 
                             strong_transform=transform_strong,
                             flist=os.path.join(data_dir, f'filelist/train_unlabeled_full.txt'))
    test_dset = BIRD30Datetset(root=os.path.join(data_dir, "images"), 
                             transform=transform_val, 
                             is_ulb=False, 
                             alg=alg,
                             flist=os.path.join(data_dir, f'filelist/test.txt'))
    eval_dset = BIRD30Datetset(root=os.path.join(data_dir, "images"), 
                             transform=transform_val, 
                             is_ulb=False, 
                             alg=alg,
                             flist=os.path.join(data_dir, f'filelist/test_closed.txt'))
    # test_data, test_targets = test_dset.data, test_dset.targets
    # test_targets[test_targets >= num_classes] = num_classes
    # seen_indices = np.where(test_targets < num_classes)[0]
    # print(len(seen_indices))
    # eval_dset = copy.deepcopy(test_dset)
    # eval_dset.data, eval_dset.targets = eval_dset.data[seen_indices], eval_dset.targets[seen_indices]

    return lb_dset, ulb_dset, eval_dset, test_dset
