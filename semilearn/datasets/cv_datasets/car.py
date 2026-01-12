import pathlib
from typing import Callable, Optional, Any, Tuple

from PIL import Image
import torch

from torchvision.datasets.utils import check_integrity,download_and_extract_archive, download_url, verify_str_arg
from torchvision.datasets.vision import VisionDataset


class StanfordCars(VisionDataset):
    """Stanford Cars Dataset

    The Cars dataset contains 16,185 images of 196 classes of cars. The data is
    split into 8,144 training images and 8,041 testing images, where each class
    has been split roughly in a 50-50 split

    .. note::

        This class needs scipy to load target files from `.mat` format.

    Args:
        root (string): Root directory of dataset
        split (string, optional): The dataset split, supports ``"train"`` (default) or ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        try:
            import scipy.io as sio
        except ImportError:
            raise RuntimeError("Scipy is not found. This dataset needs to have scipy installed: pip install scipy")

        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = pathlib.Path(root) / "Stanford_Cars"
        devkit = self._base_folder / "devkit"

        if self._split == "train":
            self._annotations_mat_path = devkit / "cars_train_annos.mat"
            self._images_base_path = self._base_folder / "cars_train"
        else:
            self._annotations_mat_path = self._base_folder / "cars_test_annos_withlabels.mat"
            self._images_base_path = self._base_folder / "cars_test"

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._samples = [
            (
                str(self._images_base_path / annotation["fname"]),
                annotation["class"] - 1,  # Original target mapping  starts from 1, hence -1
            )
            for annotation in sio.loadmat(self._annotations_mat_path, squeeze_me=True)["annotations"]
        ]

        self.classes = sio.loadmat(str(devkit / "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.class_names_str = self.classes

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Returns pil_image and class_id for given index"""
        image_path, target = self._samples[idx]
        pil_image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            pil_image = self.transform(pil_image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return pil_image, target


    def download(self) -> None:
        if self._check_exists():
            return

        # For anonymous review, dataset download URLs are removed
        raise NotImplementedError("Please download the dataset manually and place it in the correct directory.")

    def _check_exists(self) -> bool:
        if not (self._base_folder / "devkit").is_dir():
            return False

        return self._annotations_mat_path.exists() and self._images_base_path.is_dir()


def examine_count(counter, name = "train"):
    print(f"in the {name} set")
    for label in counter:
        print(label, counter[label])

import os
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from torchvision import transforms
import math
import numpy as np
from PIL import Image
import random

mean, std = {}, {}
mean['car']=[0.485, 0.456, 0.406]
std['car']=[0.229, 0.224, 0.225]

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def make_dataset_from_list(path, flist):
    path = os.path.join(path, "Stanford_Cars/filelist", flist)
    with open(path) as f:
        lines = f.readlines()
        imgs = [line.split(' ')[0] for line in lines]
        targets = [int(line.split(' ')[1].strip()) for line in lines]
        imgs = np.array(imgs)
        targets = np.array(targets)
    return imgs, targets
 
def make_train_set(args, path):
    dataset = StanfordCars(path, split="train", download=False)
    dataset.targets = [target for _, target in dataset] 
    classes = list(set(dataset.targets))

    # split id classes and ood classes
    total_classes = {i: [] for i in classes}
    id_classes = {i: [] for i in classes if i % 2 != 0}
    ood_classes = {i: [] for i in classes if i not in id_classes.keys()}
    for img, label in zip(dataset._samples, dataset.targets):
        if label % 2 != 0:
            id_classes[label].append(img[0]) 
        else:
            ood_classes[label].append(img[0]) 
    
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
    lb_txt_path = os.path.join(path, 'Stanford_Cars/filelist', f'train_labeled_{args.num_labels}.txt')
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
    ulb_txt_path = os.path.join(path, 'Stanford_Cars/filelist', f'train_unlabeled_full.txt')
    if not os.path.exists(os.path.dirname(ulb_txt_path)):
        os.makedirs(os.path.dirname(ulb_txt_path))
    with open(ulb_txt_path, 'w') as f:
        for img, label in zip(ulb_imgs, ulb_targets):
            f.write(img + ' ' + str(label) + '\n')
    
    return lb_imgs, lb_targets, ulb_imgs, ulb_targets

def make_test_set(args, path):
    dataset = StanfordCars(path, split='test', download=False)
    dataset.targets = [target for _, target in dataset] 
    classes = list(set(dataset.targets))
    test_id_classes = {i: [] for i in classes if i % 2 != 0}
    test_full_classes = {i: [] for i in classes}

    for img, label in zip(dataset._samples, dataset.targets):
        test_full_classes[label].append(img[0])
        if label % 2 != 0:
            test_id_classes[label].append(img[0])

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
    test_lb_txt_path = os.path.join(path, 'Stanford_Cars/filelist', f'test_closed.txt')
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
    test_ulb_txt_path = os.path.join(path, 'Stanford_Cars/filelist', f'test_full.txt')
    if not os.path.exists(os.path.dirname(test_ulb_txt_path)):
        os.makedirs(os.path.dirname(test_ulb_txt_path))
    with open(test_ulb_txt_path, 'w') as f:
        for img, label in zip(test_full_imgs, test_full_targets):
            f.write(img + ' ' + str(label) + '\n')

    return test_lb_imgs, test_lb_targets, test_full_imgs, test_full_targets

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class CAR196Datetset(BasicDataset):
    def __init__(self, root, transform, is_ulb, is_closed, args, alg, strong_transform=None, flist=None, train=True):
        super(CAR196Datetset, self).__init__(alg=alg, data=None, is_ulb=is_ulb,
                                                transform=transform, strong_transform=strong_transform)
        self.root = root
        self.train = train
        if flist is not None:
            print("loding from flist")
            self.data = make_dataset_from_list(self.root, flist)
            imgs, targets = self.data
        else:
            if self.train:
                if is_ulb:
                    _, _, imgs, targets = make_train_set(args, self.root)
                else:
                    imgs, targets, _, _ = make_train_set(args, self.root)
            else:
                if is_closed:
                    imgs, targets, _, _ = make_test_set(args, self.root)
                else:
                    _, _, imgs, targets = make_test_set(args, self.root)
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

def get_car196(args, alg):
    img_size = args.img_size
    crop_ratio = args.crop_ratio
    transform_weak = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['car'], std['car'])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['car'], std['car'])
    ])
    # transform_weak = RandomCrop(img_size, crop_ratio)
    # transform_strong = AugRandomCrop(img_size, crop_ratio)
    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean['car'], std['car'])
    ])

    path = os.path.join(args.data_dir)

    lb_dset = CAR196Datetset(root=path, 
                            flist=f"l_train_{args.num_labels}.txt",
                             train=False,
                             transform=transform_weak, 
                             is_ulb=False, 
                             is_closed=False, 
                             args=args,
                             alg=alg
                             )
    ulb_dset = CAR196Datetset(root=path, 
                            flist=f"u_train_{args.num_labels}.txt",
                             train=False,
                             transform=transform_weak, 
                             strong_transform=transform_strong,
                             is_ulb=True, 
                             is_closed=False, 
                             args=args,
                             alg=alg
                             )
    # make OOD dectetion dataset
    id_dset = CAR196Datetset(root=path, 
                            flist="u_train_in_1960.txt",
                            train=False,
                            transform=transform_val, 
                            is_ulb=False, 
                            is_closed=True, 
                            args=args,
                            alg=alg
                            )
    ood_dset = CAR196Datetset(root=path, 
                            # flist="test_ood.txt",
                            # flist="u_train_out_490.txt",
                            flist="u_train_out_1960.txt",
                            train=False,
                            transform=transform_val, 
                            is_ulb=False, 
                            is_closed=True, 
                            args=args,
                            alg=alg
                            )
    
    eval_dset = CAR196Datetset(root=path, 
                            flist="test_all_diy.txt",
                            train=False,
                            transform=transform_val, 
                            is_ulb=False, 
                            is_closed=True, 
                            args=args,
                            alg=alg
                            )
    test_dset = CAR196Datetset(root=path, 
                            flist="test_all.txt",
                            train=False,
                            transform=transform_val, 
                            is_ulb=False, 
                            is_closed=False, 
                            args=args,
                            alg=alg
                            )

    return lb_dset, ulb_dset, eval_dset, test_dset, id_dset, ood_dset

def find_car_classnames(args):
    path = os.path.join(args.data_dir)
    classnames_path = os.path.join(path, "Stanford_Cars", "filelist", "classnames.txt")
    
    # 优先使用 filelist 中的 classnames.txt
    if os.path.exists(classnames_path):
        with open(classnames_path, 'r') as f:
            lb_cls_list = [line.strip() for line in f.readlines() if line.strip()]
        return lb_cls_list
    
    # 如果 txt 文件不存在，从 cars_meta.mat 读取
    import scipy.io as sio
    devkit_path = os.path.join(path, "Stanford_Cars", "devkit", "cars_meta.mat")
    class_names = sio.loadmat(devkit_path, squeeze_me=True)["class_names"].tolist()
    lb_cls_list = []
    for idx, cls in enumerate(class_names):
        if idx % 2 == 0:
            lb_cls_list.append(cls)
    return lb_cls_list