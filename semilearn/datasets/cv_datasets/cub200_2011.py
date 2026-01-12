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
mean['cub']=[0.485, 0.456, 0.406]
std['cub']=[0.229, 0.224, 0.225]

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'

    def __init__(self, root, train=True, transform=None, loader=default_loader):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        self._load_metadata()

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

        class_names = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'classes.txt'),
                                         sep=' ', names=['class_id', 'target'])
        self.class_names_str = [name.split(".")[1].replace('_', ' ') for name in class_names.target]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    

def make_dataset_from_list(path, flist):
    filelist_path = os.path.join(path, 'CUB_200_2011/filelist', flist)
    path = os.path.join(path, 'CUB_200_2011/images')
    with open(filelist_path) as f:
        lines = f.readlines()
        # imgs = [line.split(' ')[0] for line in lines]
        imgs = [os.path.join(path, line.split(' ')[0]) for line in lines]
        targets = [int(line.split(' ')[1].strip()) for line in lines]
        imgs = np.array(imgs)
        targets = np.array(targets)
    return imgs, targets

def make_train_set(args, path):
    dataset = Cub2011(path, train=True)
    dataset.targets = [target for _, target in dataset] 
    classes = list(set(dataset.targets))
   
    # split id classes and ood classes
    total_classes = {i: [] for i in classes}
    id_classes = {i: [] for i in classes if i % 2 != 0}
    ood_classes = {i: [] for i in classes if i not in id_classes.keys()}
    for img, label in zip(dataset.data.iloc, dataset.targets):
        if label % 2 != 0:
            id_classes[label].append(os.path.join(path, 'CUB_200_2011/images', img[1])) 
        else:
            ood_classes[label].append(os.path.join(path, 'CUB_200_2011/images', img[1]))
    
    # split labled data from id classes for train set
    id_labeled_classes = {i: [] for i in id_classes.keys()}
    unlabeled_classes = {i: [] for i in classes}
    
    for label in id_classes.keys():
        selected_imgs = random.sample(id_classes[label], int(args.num_labels / len(id_labeled_classes.keys()))) 
        remaining_imgs = [img for img in id_classes[label] if img not in selected_imgs]
        id_labeled_classes[label].extend(selected_imgs)
        unlabeled_classes[label].extend(remaining_imgs)
    for label in ood_classes.keys():
        unlabeled_classes[label].extend(ood_classes[label])
    
    lb_imgs = []
    lb_targets = []
    i = 0 # sort the labeles of id class from 0
    for label in id_labeled_classes.keys():
        for img in id_labeled_classes[label]:
            lb_imgs.append(img)
            lb_targets.append(i)
        i += 1
    lb_imgs = np.array(lb_imgs)
    lb_targets = np.array(lb_targets)
    # make the imgs and labels of labeled data to txt
    lb_txt_path = os.path.join(path, 'CUB_200_2011/filelist', f'train_labeled_{args.num_labels}.txt')
    if not os.path.exists(os.path.dirname(lb_txt_path)):
        os.makedirs(os.path.dirname(lb_txt_path))
    with open(lb_txt_path, 'w') as f:
        for img, label in zip(lb_imgs, lb_targets):
            f.write(img + ' ' + str(label) + '\n')

    ulb_imgs = []
    ulb_targets = []
    i = 0
    j = args.num_classes # sort the labeles of ood class from 60
    for label in unlabeled_classes.keys():
        if label % 2 != 0:
            for img in unlabeled_classes[label]:
                ulb_imgs.append(img)
                ulb_targets.append(i)
            i += 1
    for label in unlabeled_classes.keys():
        if label % 2 == 0:
            for img in unlabeled_classes[label]:
                ulb_imgs.append(img)
                ulb_targets.append(j)
            j += 1

    ulb_imgs = np.array(ulb_imgs)
    ulb_targets = np.array(ulb_targets)

    # make imgs and labels of unlabeled data to txt
    ulb_txt_path = os.path.join(path, 'CUB_200_2011/filelist', f'train_unlabeled_full.txt')
    if not os.path.exists(os.path.dirname(ulb_txt_path)):
        os.makedirs(os.path.dirname(ulb_txt_path))
    with open(ulb_txt_path, 'w') as f:
        for img, label in zip(ulb_imgs, ulb_targets):
            f.write(img + ' ' + str(label) + '\n')
    
    return lb_imgs, lb_targets, ulb_imgs, ulb_targets

def make_test_set(args, path):
    dataset = Cub2011(path, train=False)
    dataset.targets = [target for _, target in dataset] 
    classes = list(set(dataset.targets))
    test_id_classes = {i: [] for i in classes if i % 2 != 0}
    test_full_classes = {i: [] for i in classes}

    for img, label in zip(dataset.data.iloc, dataset.targets):
        test_full_classes[label].append(os.path.join(path, 'CUB_200_2011/images', img[1])) 
        if label % 2 != 0:
            test_id_classes[label].append(os.path.join(path, 'CUB_200_2011/images', img[1])) 

    test_lb_imgs = []
    test_lb_targets = []
    i = 0
    for label in test_id_classes.keys():
        for img in test_id_classes[label]:
            test_lb_imgs.append(img)
            test_lb_targets.append(i)
        i += 1
    test_lb_imgs = np.array(test_lb_imgs)
    test_lb_targets = np.array(test_lb_targets)

    # make imgs and labels of closed test data to txt
    test_lb_txt_path = os.path.join(path, 'CUB_200_2011/filelist', f'test_closed.txt')
    if not os.path.exists(os.path.dirname(test_lb_txt_path)):
        os.makedirs(os.path.dirname(test_lb_txt_path))
    with open(test_lb_txt_path, 'w') as f:
        for img, label in zip(test_lb_imgs, test_lb_targets):
            f.write(img + ' ' + str(label) + '\n')

    test_full_imgs = []
    test_full_targets = []
    i = 0
    j = args.num_classes
    for label in test_full_classes.keys():
        if label % 2 != 0:
            for img in test_full_classes[label]:
                test_full_imgs.append(img)
                test_full_targets.append(i)
            i += 1
    for label in test_full_classes.keys():
        if label % 2 == 0:
            for img in test_full_classes[label]:
                test_full_imgs.append(img)
                test_full_targets.append(j)
            j += 1
    test_full_imgs = np.array(test_full_imgs)
    test_full_targets = np.array(test_full_targets)
    # make imgs and labels of full test data to txt
    test_ulb_txt_path = os.path.join(path, 'CUB_200_2011/filelist', f'test_full.txt')
    if not os.path.exists(os.path.dirname(test_ulb_txt_path)):
        os.makedirs(os.path.dirname(test_ulb_txt_path))
    with open(test_ulb_txt_path, 'w') as f:
        for img, label in zip(test_full_imgs, test_full_targets):
            f.write(img + ' ' + str(label) + '\n')

    return test_lb_imgs, test_lb_targets, test_full_imgs, test_full_targets

class CUB200Datetset(BasicDataset):
    def __init__(self, root, transform, is_ulb, is_closed, args, alg, strong_transform=None, flist=None, train=True):
        super(CUB200Datetset, self).__init__(alg=alg, data=None, is_ulb=is_ulb,
                                                transform=transform, strong_transform=strong_transform)
        self.root = root
        self.train = train
        if flist is not None:
            print('##### loading from flist #####')
            imgs, targets = make_dataset_from_list(self.root, flist)
    
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

def get_cub200(args, alg):
    img_size = args.img_size
    crop_ratio = args.crop_ratio
    transform_weak = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['cub'], std['cub'])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['cub'], std['cub'])
    ])
  
    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean['cub'], std['cub'])
    ])

    path = os.path.join(args.data_dir)

    lb_dset = CUB200Datetset(root=path, 
                             flist=f'l_train_{args.num_labels}.txt',
                             train=True,
                             transform=transform_weak, 
                             is_ulb=False, 
                             is_closed=False, 
                             args=args,
                             alg=alg
                             )
    ulb_dset = CUB200Datetset(root=path, 
                              flist=f'u_train_{args.num_labels}.txt',
                             train=True,
                             transform=transform_weak, 
                             strong_transform=transform_strong,
                             is_ulb=True, 
                             is_closed=False, 
                             args=args,
                             alg=alg
                             )
    # make OOD dectetion dataset
    id_dset = CUB200Datetset(root=path, 
                            flist="test_id.txt",
                            train=False,
                            transform=transform_val, 
                            is_ulb=False, 
                            is_closed=True, 
                            args=args,
                            alg=alg
                            )
    ood_dset = CUB200Datetset(root=path, 
                            flist="test_ood.txt",
                            train=False,
                            transform=transform_val, 
                            is_ulb=False, 
                            is_closed=True, 
                            args=args,
                            alg=alg
                            )
    eval_dset = CUB200Datetset(root=path, 
                            flist='test_full.txt',
                            train=False,
                            transform=transform_val, 
                            is_ulb=False, 
                            is_closed=True, 
                            args=args,
                            alg=alg
                            )
    test_dset = CUB200Datetset(root=path,
                            flist='test.txt', 
                            train=False,
                            transform=transform_val, 
                            is_ulb=False, 
                            is_closed=False, 
                            args=args,
                            alg=alg
                            )

    return lb_dset, ulb_dset, eval_dset, test_dset, id_dset, ood_dset

def find_cub_classnames(args):
    path = os.path.join(args.data_dir)  
    dataset = Cub2011(path, train=False)
    classnames = dataset.class_names_str
    class_to_idx = {cls: i for i, cls in enumerate(classnames)}
    lb_cls_list = []
    for cls, idx in class_to_idx.items():
        if idx % 2 == 0:
            lb_cls_list.append(cls)
    return lb_cls_list