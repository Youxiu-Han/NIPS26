import numpy as np
import os
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision.datasets.utils import extract_archive
import os
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from torchvision import transforms
import math
import numpy as np
from PIL import Image
import random

mean, std = {}, {}
mean['aircraft']=[0.485, 0.456, 0.406]
std['aircraft']=[0.229, 0.224, 0.225]

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

class FGVCAircraft(VisionDataset):
    download_url_prefix = ''  # For anonymous review, we removed all URLs and identifiers in the code to prevent unintended disclosure of author identity.
    class_types = ("variant", "family", "manufacturer")
    splits = ("train", "val", "trainval", "test")
    img_folder = os.path.join("fgvc-aircraft-2013b", "data", "images")

    def __init__(
        self,
        root,
        train=True,
        class_type="variant",
        transform=None,
        target_transform=None,
        download=False,
    ):
        super(FGVCAircraft, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        split = "trainval" if train else "test"
        if split not in self.splits:
            raise ValueError(
                'Split "{}" not found. Valid splits are: {}'.format(
                    split,
                    ", ".join(self.splits),
                )
            )
        if class_type not in self.class_types:
            raise ValueError(
                'Class type "{}" not found. Valid class types are: {}'.format(
                    class_type,
                    ", ".join(self.class_types),
                )
            )

        self.class_type = class_type
        self.split = split
        self.classes_file = os.path.join(
            self.root,
            "fgvc-aircraft-2013b",
            "data",
            "images_%s_%s.txt" % (self.class_type, self.split),
        )

        if download:
            self.download()

        (image_ids, targets, classes, class_to_idx) = self.find_classes()
        samples = self.make_dataset(image_ids, targets)

        self.loader = default_loader

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def _check_exists(self):
        return os.path.exists(
            os.path.join(self.root, self.img_folder)
        ) and os.path.exists(self.classes_file)

    def download(self):
        if self._check_exists():
            return

        # prepare to download data to PARENT_DIR/fgvc-aircraft-2013.tar.gz
        print("Downloading %s..." % self.url)
        tar_name = self.url.rpartition("/")[-1]
        download_url(self.url, root=self.root, filename=tar_name)
        tar_path = os.path.join(self.root, tar_name)
        print("Extracting %s..." % tar_path)
        extract_archive(tar_path)
        print("Done!")

    def find_classes(self):
        # read classes file, separating out image IDs and class names
        image_ids = []
        targets = []
        with open(self.classes_file, "r") as f:
            for line in f:
                line = line.strip()
                split_line = line.split(" ")
                image_ids.append(split_line[0])
                targets.append(" ".join(split_line[1:]))

        # index class names
        classes = np.unique(targets)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        targets = [class_to_idx[c] for c in targets]

        return image_ids, targets, classes, class_to_idx

    def make_dataset(self, image_ids, targets):
        assert len(image_ids) == len(targets)
        images = []
        for i in range(len(image_ids)):
            item = (
                os.path.join(self.root, self.img_folder, "%s.jpg" % image_ids[i]),
                targets[i],
            )
            images.append(item)
        return images

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
def make_dataset_from_list(path, flist):
    flist = os.path.join(path, "fgvc-aircraft-2013b/filelist", flist)
    with open(flist) as f:
        lines = f.readlines()
        imgs = [line.split(' ')[0] for line in lines]
        targets = [int(line.split(' ')[1].strip()) for line in lines]
        imgs = np.array(imgs)
        targets = np.array(targets)
    return imgs, targets

def make_train_set(args, path):
    dataset = FGVCAircraft(path, train=True)
    dataset.targets = [target for _, target in dataset] 
    classes = list(set(dataset.targets))
    # print('classes', classes) 
    # split id classes and ood classes
    total_classes = {i: [] for i in classes}
    id_classes = {i: [] for i in classes if i % 2 != 0}
    ood_classes = {i: [] for i in classes if i not in id_classes.keys()}
    for img, label in zip(dataset.samples, dataset.targets):
        if label % 2 != 0:
            id_classes[label].append(img[0]) 
        else:
            ood_classes[label].append(img[0]) 
    # print('id_classes', id_classes[1])
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
    lb_txt_path = os.path.join(path, 'fgvc-aircraft-2013b/filelist', f'train_labeled_{args.num_labels}.txt')
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
    ulb_txt_path = os.path.join(path, 'fgvc-aircraft-2013b/filelist', f'train_unlabeled_full.txt')
    if not os.path.exists(os.path.dirname(ulb_txt_path)):
        os.makedirs(os.path.dirname(ulb_txt_path))
    with open(ulb_txt_path, 'w') as f:
        for img, label in zip(ulb_imgs, ulb_targets):
            f.write(img + ' ' + str(label) + '\n')
    
    return lb_imgs, lb_targets, ulb_imgs, ulb_targets

def make_test_set(args, path):
    dataset = FGVCAircraft(path, train=False)
    dataset.targets = [target for _, target in dataset] 
    classes = list(set(dataset.targets))
    test_id_classes = {i: [] for i in classes if i % 2 != 0}
    test_full_classes = {i: [] for i in classes}

    for img, label in zip(dataset.samples, dataset.targets):
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
    test_lb_txt_path = os.path.join(path, 'fgvc-aircraft-2013b/filelist', f'test_closed.txt')
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
    test_ulb_txt_path = os.path.join(path, 'fgvc-aircraft-2013b/filelist', f'test_full.txt')
    if not os.path.exists(os.path.dirname(test_ulb_txt_path)):
        os.makedirs(os.path.dirname(test_ulb_txt_path))
    with open(test_ulb_txt_path, 'w') as f:
        for img, label in zip(test_full_imgs, test_full_targets):
            f.write(img + ' ' + str(label) + '\n')

    return test_lb_imgs, test_lb_targets, test_full_imgs, test_full_targets



class AIRCRAFT100Datetset(BasicDataset):
    def __init__(self, root, transform, is_ulb, is_closed, args, alg, strong_transform=None, flist=None, train=True):
        super(AIRCRAFT100Datetset, self).__init__(alg=alg, data=None, is_ulb=is_ulb,
                                                transform=transform, strong_transform=strong_transform)
        self.root = root
        self.train = train
        if flist is not None:
            print('loading from flist')
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
        
def get_aircraft100(args, alg):
    img_size = args.img_size
    crop_ratio = args.crop_ratio
    transform_weak = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['aircraft'], std['aircraft'])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['aircraft'], std['aircraft'])
    ])
    # transform_weak = RandomCrop(img_size, crop_ratio)
    # transform_strong = AugRandomCrop(img_size, crop_ratio)
    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean['aircraft'], std['aircraft'])
    ])

    path = os.path.join(args.data_dir)

    lb_dset = AIRCRAFT100Datetset(root=path, 
                            flist=f"l_train_{args.num_labels}.txt",
                             train=True,
                             transform=transform_weak, 
                             is_ulb=False, 
                             is_closed=False, 
                             args=args,
                             alg=alg
                             )
    ulb_dset = AIRCRAFT100Datetset(root=path, 
                            flist=f"u_train_{args.num_labels}.txt",
                             train=True,
                             transform=transform_weak, 
                             strong_transform=transform_strong,
                             is_ulb=True, 
                             is_closed=False, 
                             args=args,
                             alg=alg
                             )
    # make OOD dectetion dataset
    id_dset = AIRCRAFT100Datetset(root=path, 
                            flist="u_train_in_250.txt",
                            train=False,
                            transform=transform_val, 
                            is_ulb=False, 
                            is_closed=True, 
                            args=args,
                            alg=alg
                            )
    ood_dset = AIRCRAFT100Datetset(root=path, 
                            flist="u_train_out_250.txt",
                            train=False,
                            transform=transform_val, 
                            is_ulb=False, 
                            is_closed=True, 
                            args=args,
                            alg=alg
                            )
    eval_dset = AIRCRAFT100Datetset(root=path, 
                            flist="test_all_diy.txt",
                            train=False,
                            transform=transform_val, 
                            is_ulb=False, 
                            is_closed=True, 
                            args=args,
                            alg=alg
                            )
    test_dset = AIRCRAFT100Datetset(root=path, 
                            flist= "test_all.txt",
                            train=False,
                            transform=transform_val, 
                            is_ulb=False, 
                            is_closed=False, 
                            args=args,
                            alg=alg
                            )

    return lb_dset, ulb_dset, eval_dset, test_dset, id_dset, ood_dset

def find_aircraft_classnames(args):
    path = os.path.join(args.data_dir)
    dataset = FGVCAircraft(path, train=False)
    class_to_idx = dataset.class_to_idx
    lb_cls_list = []
    for cls, idx in class_to_idx.items():
        if idx % 2 == 0:
            lb_cls_list.append(cls)
    return lb_cls_list
