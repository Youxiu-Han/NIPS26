
import os 
import torch
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from torchvision import transforms
import math
import numpy as np
from PIL import Image
import random
from semilearn.datasets.dataloaders import StanfordDogs

# The num of total train labeles is 12000; the num of total test labels is 8580.
# The num of total labels is 200 (0~199).
# (1,3,7,...,199) is the id classes, (0,2,4,...,198) is the ood classes.

mean, std = {}, {}
# mean['dog']=[0.485, 0.456, 0.406]
# std['dog']=[0.229, 0.224, 0.225]
mean['dog']=[0.48145466, 0.4578275, 0.40821073]
std['dog']=[0.26862954, 0.26130258, 0.27577711]

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
    
def make_dataset_from_list(path, flist):
    path = os.path.join(path, "Stanford_Dogs/filelist", flist)
    with open(path) as f:
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
    
def make_train_set(args, path):
    custum_path = os.path.join(path, 'Images')
    dataset = StanfordDogs(path, train=True, download=False)
    dataset.targets = [target for _, target in dataset] 
    classes = list(set(dataset.targets))
    
    # split id classes and ood classes
    total_classes = {i: [] for i in classes}
    id_classes = {i: [] for i in classes if i % 2 != 0}
    ood_classes = {i: [] for i in classes if i not in id_classes.keys()}
    for img, label in zip(dataset._flat_breed_images, dataset.targets):
        if label % 2 != 0:
            id_classes[label].append(os.path.join(custum_path, img[0])) 
        else:
            ood_classes[label].append(os.path.join(custum_path, img[0])) 
   
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
    lb_txt_path = os.path.join(path, 'filelist', f'train_labeled_{args.num_labels}.txt')
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
    ulb_txt_path = os.path.join(path, 'filelist', f'train_unlabeled_full.txt')
    if not os.path.exists(os.path.dirname(ulb_txt_path)):
        os.makedirs(os.path.dirname(ulb_txt_path))
    with open(ulb_txt_path, 'w') as f:
        for img, label in zip(ulb_imgs, ulb_targets):
            f.write(img + ' ' + str(label) + '\n')
    
    return lb_imgs, lb_targets, ulb_imgs, ulb_targets

def make_test_set(args, path):
    custum_path = os.path.join(path, 'Images')
    dataset = StanfordDogs(path, train=False, download=False)
    dataset.targets = [target for _, target in dataset] 
    classes = list(set(dataset.targets))
    test_id_classes = {i: [] for i in classes if i % 2 != 0}
    test_full_classes = {i: [] for i in classes}

    for img, label in zip(dataset._flat_breed_images, dataset.targets):
        test_full_classes[label].append(os.path.join(custum_path, img[0]))
        if label % 2 != 0:
            test_id_classes[label].append(os.path.join(custum_path, img[0]))

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
    test_lb_txt_path = os.path.join(path, 'filelist', f'test_closed.txt')
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
    test_ulb_txt_path = os.path.join(path, 'filelist', f'test_full.txt')
    if not os.path.exists(os.path.dirname(test_ulb_txt_path)):
        os.makedirs(os.path.dirname(test_ulb_txt_path))
    with open(test_ulb_txt_path, 'w') as f:
        for img, label in zip(test_full_imgs, test_full_targets):
            f.write(img + ' ' + str(label) + '\n')

    return test_lb_imgs, test_lb_targets, test_full_imgs, test_full_targets

def find_dog_classnames(args):
    selected_classes = []
    classes = ["Chihuaha",
                        "Japanese Spaniel",
                        "Maltese Dog",
                        "Pekinese",
                        "Shih-Tzu",
                        "Blenheim Spaniel",
                        "Papillon",
                        "Toy Terrier",
                        "Rhodesian Ridgeback",
                        "Afghan Hound",
                        "Basset Hound",
                        "Beagle",
                        "Bloodhound",
                        "Bluetick",
                        "Black-and-tan Coonhound",
                        "Walker Hound",
                        "English Foxhound",
                        "Redbone",
                        "Borzoi",
                        "Irish Wolfhound",
                        "Italian Greyhound",
                        "Whippet",
                        "Ibizian Hound",
                        "Norwegian Elkhound",
                        "Otterhound",
                        "Saluki",
                        "Scottish Deerhound",
                        "Weimaraner",
                        "Staffordshire Bullterrier",
                        "American Staffordshire Terrier",
                        "Bedlington Terrier",
                        "Border Terrier",
                        "Kerry Blue Terrier",
                        "Irish Terrier",
                        "Norfolk Terrier",
                        "Norwich Terrier",
                        "Yorkshire Terrier",
                        "Wirehaired Fox Terrier",
                        "Lakeland Terrier",
                        "Sealyham Terrier",
                        "Airedale",
                        "Cairn",
                        "Australian Terrier",
                        "Dandi Dinmont",
                        "Boston Bull",
                        "Miniature Schnauzer",
                        "Giant Schnauzer",
                        "Standard Schnauzer",
                        "Scotch Terrier",
                        "Tibetan Terrier",
                        "Silky Terrier",
                        "Soft-coated Wheaten Terrier",
                        "West Highland White Terrier",
                        "Lhasa",
                        "Flat-coated Retriever",
                        "Curly-coater Retriever",
                        "Golden Retriever",
                        "Labrador Retriever",
                        "Chesapeake Bay Retriever",
                        "German Short-haired Pointer",
                        "Vizsla",
                        "English Setter",
                        "Irish Setter",
                        "Gordon Setter",
                        "Brittany",
                        "Clumber",
                        "English Springer Spaniel",
                        "Welsh Springer Spaniel",
                        "Cocker Spaniel",
                        "Sussex Spaniel",
                        "Irish Water Spaniel",
                        "Kuvasz",
                        "Schipperke",
                        "Groenendael",
                        "Malinois",
                        "Briard",
                        "Kelpie",
                        "Komondor",
                        "Old English Sheepdog",
                        "Shetland Sheepdog",
                        "Collie",
                        "Border Collie",
                        "Bouvier des Flandres",
                        "Rottweiler",
                        "German Shepard",
                        "Doberman",
                        "Miniature Pinscher",
                        "Greater Swiss Mountain Dog",
                        "Bernese Mountain Dog",
                        "Appenzeller",
                        "EntleBucher",
                        "Boxer",
                        "Bull Mastiff",
                        "Tibetan Mastiff",
                        "French Bulldog",
                        "Great Dane",
                        "Saint Bernard",
                        "Eskimo Dog",
                        "Malamute",
                        "Siberian Husky",
                        "Affenpinscher",
                        "Basenji",
                        "Pug",
                        "Leonberg",
                        "Newfoundland",
                        "Great Pyrenees",
                        "Samoyed",
                        "Pomeranian",
                        "Chow",
                        "Keeshond",
                        "Brabancon Griffon",
                        "Pembroke",
                        "Cardigan",
                        "Toy Poodle",
                        "Miniature Poodle",
                        "Standard Poodle",
                        "Mexican Hairless",
                        "Dingo",
                        "Dhole",
                        "African Hunting Dog"]
    for i in range(len(classes)):
        if i % 2 == 0:
            selected_classes.append(classes[i])
    return selected_classes


class DOG120Datetset(BasicDataset):
    def __init__(self, root, transform, is_ulb, is_closed, args, alg, strong_transform=None, flist=None, train=True):
        super(DOG120Datetset, self).__init__(alg=alg, data=None, is_ulb=is_ulb,
                                                transform=transform, strong_transform=strong_transform)
       
        self.root = root
        self.train = train

        if flist is not None:
            print("loading from flist")
            imgs, targets = make_dataset_from_list(self.root, flist)
        else:
            # print('please provide flist')
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


class RandomCrop(object):
    def __init__(self, img_size, crop_ratio, n_crop=256):
        normalize = transforms.Normalize(mean['dog'], std['dog'])  # for CLIP
        self.n_crop = n_crop
        self.random_crop = transforms.Compose([
            # transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

    def __call__(self, x):
        views = [self.random_crop(x).unsqueeze(dim=0) for _ in range(self.n_crop)]
        views = torch.cat(views, dim=0)
        return views


class AugRandomCrop(object):
    def __init__(self, img_size, crop_ratio, n_crop=256):
        normalize = transforms.Normalize(mean['dog'], std['dog'])  # for CLIP
        self.n_crop = n_crop
        self.random_crop = transforms.Compose([
            # transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            RandAugment(3, 10),
            transforms.ToTensor(),
            normalize
        ])

    def __call__(self, x):
        views = [self.random_crop(x).unsqueeze(dim=0) for _ in range(self.n_crop)]
        views = torch.cat(views, dim=0)
        return views
def get_dog120(args, alg):
    img_size = args.img_size
    crop_ratio = args.crop_ratio
    transform_weak = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['dog'], std['dog'])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['dog'], std['dog'])
    ])
    # transform_weak = RandomCrop(img_size, crop_ratio)
    # transform_strong = AugRandomCrop(img_size, crop_ratio)
    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean['dog'], std['dog'])
    ])

    path = os.path.join(args.data_dir)

    lb_dset = DOG120Datetset(root=path, 
                             flist=f"l_train_{args.num_labels}.txt",
                             train=True,
                             transform=transform_weak, 
                             is_ulb=False, 
                             is_closed=False, 
                             args=args,
                             alg=alg
                             )
    ulb_dset = DOG120Datetset(root=path,
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
    id_dset = DOG120Datetset(root=path, 
                            flist="u_train_in_1200.txt",
                            train=False,
                            transform=transform_val, 
                            is_ulb=False, 
                            is_closed=True, 
                            args=args,
                            alg=alg
                            )
    ood_dset = DOG120Datetset(root=path, 
                            flist="u_train_out_1200.txt",
                            train=False,
                            transform=transform_val, 
                            is_ulb=False, 
                            is_closed=True, 
                            args=args,
                            alg=alg
                            )
    eval_dset = DOG120Datetset(root=path, 
                            flist="test_all_diy.txt",
                            train=False,
                            transform=transform_val, 
                            is_ulb=False, 
                            is_closed=True, 
                            args=args,
                            alg=alg
                            )
    test_dset = DOG120Datetset(root=path, 
                            flist="test_all.txt",
                            train=False,
                            transform=transform_val, 
                            is_ulb=False, 
                            is_closed=False, 
                            args=args,
                            alg=alg
                            )
    return lb_dset, ulb_dset, eval_dset, test_dset, id_dset, ood_dset

import os
import scipy.io
from os.path import join
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, list_dir


class StanfordDogs(VisionDataset):
    download_url_prefix = ''  # For anonymous review, we removed all URLs and identifiers in the code to prevent unintended disclosure of author identity.

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(StanfordDogs, self).__init__(root, transform=transform, target_transform=target_transform)

        self.loader = default_loader
        self.train = train

        if download:
            self.download()

        split = self.load_split()

        self.images_folder = join(self.root, 'Images')
        self.annotations_folder = join(self.root, 'Annotation')
        self._breeds = list_dir(self.images_folder)

        self._breed_images = [(annotation + '.jpg', idx) for annotation, idx in split]

        self._flat_breed_images = self._breed_images

    def __len__(self):
        return len(self._flat_breed_images)

    def __getitem__(self, index):
        image_name, target = self._flat_breed_images[index]
        image_path = join(self.images_folder, image_name)
        image = self.loader(image_path)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def download(self):
        import tarfile

        if os.path.exists(join(self.root, 'Images')) and os.path.exists(join(self.root, 'Annotation')):
            if len(os.listdir(join(self.root, 'Images'))) == len(os.listdir(join(self.root, 'Annotation'))) == 120:
                print('Files already downloaded and verified')
                return

        for filename in ['images', 'annotation', 'lists']:
            tar_filename = filename + '.tar'
            url = self.download_url_prefix + '/' + tar_filename
            download_url(url, self.root, tar_filename, None)
            print('Extracting downloaded file: ' + join(self.root, tar_filename))
            with tarfile.open(join(self.root, tar_filename), 'r') as tar_file:
                tar_file.extractall(self.root)
            os.remove(join(self.root, tar_filename))

    def load_split(self):
        if self.train:
            split = scipy.io.loadmat(join(self.root, 'train_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(join(self.root, 'train_list.mat'))['labels']
        else:
            split = scipy.io.loadmat(join(self.root, 'test_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(join(self.root, 'test_list.mat'))['labels']
        
        split = [item[0][0] for item in split]
        labels = [item[0] - 1 for item in labels]
    
        return list(zip(split, labels))

    def stats(self):
        counts = {}
        for index in range(len(self._flat_breed_images)):
            image_name, target_class = self._flat_breed_images[index]
            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1

        print("%d samples spanning %d classes (avg %f per class)" % (len(self._flat_breed_images), len(counts.keys()),
                                                                     float(len(self._flat_breed_images)) / float(
                                                                         len(counts.keys()))))

        return counts




    

