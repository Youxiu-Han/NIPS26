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
mean['aves']=[0.485, 0.456, 0.406]
std['aves']=[0.229, 0.224, 0.225]

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
    filelist_path = os.path.join(path, 'semi_aves/filelist', flist)
    path = os.path.join(path, 'semi_aves')
    with open(filelist_path) as f:
        lines = f.readlines()
        # imgs = [line.split(' ')[0] for line in lines]
        imgs = [os.path.join(path, line.split(' ')[0]) for line in lines]
        targets = [int(line.split(' ')[1].strip()) for line in lines]
        imgs = np.array(imgs)
        targets = np.array(targets)
    return imgs, targets

class SEMIAVESDatetset(BasicDataset):
    def __init__(self, root, transform, is_ulb, is_closed, args, alg, strong_transform=None, flist=None, train=True):
        super(SEMIAVESDatetset, self).__init__(alg=alg, data=None, is_ulb=is_ulb,
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

def get_semiaves200(args, alg):
    img_size = args.img_size
    crop_ratio = args.crop_ratio
    transform_weak = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['aves'], std['aves'])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['aves'], std['aves'])
    ])
    # transform_weak = RandomCrop(img_size, crop_ratio)
    # transform_strong = AugRandomCrop(img_size, crop_ratio)
    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean['aves'], std['aves'])
    ])

    path = os.path.join(args.data_dir)

    lb_dset = SEMIAVESDatetset(root=path, 
                             flist='l_train_val.txt',
                             train=True,
                             transform=transform_weak, 
                             is_ulb=False, 
                             is_closed=False, 
                             args=args,
                             alg=alg
                             )
    ulb_dset = SEMIAVESDatetset(root=path, 
                              flist='u_train.txt',
                             train=True,
                             transform=transform_weak, 
                             strong_transform=transform_strong,
                             is_ulb=True, 
                             is_closed=False, 
                             args=args,
                             alg=alg
                             )
    eval_dset = SEMIAVESDatetset(root=path, 
                            flist='val.txt',
                            train=False,
                            transform=transform_val, 
                            is_ulb=False, 
                            is_closed=True, 
                            args=args,
                            alg=alg
                            )
    test_dset = SEMIAVESDatetset(root=path,
                            flist='test.txt', 
                            train=False,
                            transform=transform_val, 
                            is_ulb=False, 
                            is_closed=False, 
                            args=args,
                            alg=alg
                            )

    return lb_dset, ulb_dset, test_dset, test_dset

def find_aves_classnames(args):
    # path = os.path.join(args.data_dir)
    # classnames_path = os.path.join(path, 'semi_aves/filelist/semi_aves_species_names.txt')
    # with open(classnames_path) as f:
    #     lines = f.readlines()
    #     classnames = [line.split(' ')[1].strip('\n') for line in lines]
    # cls_list= []
    # for i in range(len(classnames)):
    #     if i % 2 ==0:
    #         cls_list.append(classnames[i])
    classnames = [
    "Charadrius_semipalmatus",
    "Tachycineta_thalassina",
    "Stelgidopteryx_serripennis",
    "Cassiculus_melanicterus",
    "Jacana_spinosa",
    "Haematopus_bachmani",
    "Phoenicopterus_ruber",
    "Phoenicurus_phoenicurus",
    "Pelecanus_occidentalis",
    "Archilochus_alexandri",
    "Pelecanus_conspicillatus",
    "Falco_tinnunculus",
    "Tachybaptus_novaehollandiae",
    "Petrochelidon_pyrrhonota",
    "Falco_peregrinus",
    "Oenanthe_oenanthe",
    "Calidris_pusilla",
    "Calidris_alpina",
    "Streptopelia_chinensis",
    "Limosa_fedoa",
    "Anthochaera_carunculata",
    "Columba_livia",
    "Phalaropus_lobatus",
    "Thalasseus_sandvicensis",
    "Larus_canus",
    "Motacilla_cinerea",
    "Aechmophorus_occidentalis",
    "Larus_glaucescens",
    "Dendrocopos_major",
    "Rissa_tridactyla",
    "Molothrus_aeneus",
    "Entomyzon_cyanotis",
    "Alopochen_aegyptiaca",
    "Lonchura_punctulata",
    "Anas_platyrhynchos",
    "Aythya_australis",
    "Dicrurus_macrocercus",
    "Anser_albifrons",
    "Buteo_regalis",
    "Haliaeetus_leucocephalus",
    "Morus_bassanus",
    "Certhia_familiaris",
    "Rostrhamus_sociabilis",
    "Cracticus_torquatus",
    "Passerina_amoena",
    "Aythya_fuligula",
    "Chondestes_grammacus",
    "Fulica_atra",
    "Mergus_merganser",
    "Egretta_thula",
    "Pyrrhula_pyrrhula",
    "Aquila_chrysaetos",
    "Setophaga_townsendi",
    "Cygnus_atratus",
    "Cygnus_buccinator",
    "Branta_canadensis",
    "Circus_hudsonius",
    "Anhinga_anhinga",
    "Buteo_jamaicensis",
    "Pica_pica",
    "Corvus_frugilegus",
    "Bonasa_umbellus",
    "Myiozetetes_similis",
    "Progne_subis",
    "Megaceryle_alcyon",
    "Molothrus_ater",
    "Piranga_rubra",
    "Vanellus_chilensis",
    "Cygnus_olor",
    "Passerculus_sandwichensis",
    "Chroicocephalus_philadelphia",
    "Setophaga_palmarum",
    "Himantopus_himantopus",
    "Setophaga_occidentalis",
    "Buteo_lineatus",
    "Zonotrichia_leucophrys",
    "Saltator_coerulescens",
    "Tyrannus_vociferans",
    "Junco_hyemalis",
    "Pitangus_sulphuratus",
    "Sylvia_communis",
    "Melanerpes_erythrocephalus",
    "Rhipidura_leucophrys",
    "Quiscalus_major",
    "Sphyrapicus_varius",
    "Sitta_europaea",
    "Cyanocorax_yncas",
    "Corvus_brachyrhynchos",
    "Sitta_canadensis",
    "Cyanistes_caeruleus",
    "Colaptes_auratus",
    "Baeolophus_inornatus",
    "Erithacus_rubecula",
    "Poecile_carolinensis",
    "Passer_domesticus",
    "Sphecotheres_vieilloti",
    "Mniotilta_varia",
    "Dryocopus_pileatus",
    "Prunella_modularis",
    "Megarynchus_pitangua",
    "Melospiza_georgiana",
    "Agelaius_phoeniceus",
    "Lanius_cristatus",
    "Catharus_ustulatus",
    "Ammodramus_savannarum",
    "Thryothorus_ludovicianus",
    "Troglodytes_aedon",
    "Setophaga_caerulescens",
    "Neochmia_temporalis",
    "Setophaga_citrina",
    "Myiodynastes_luteiventris",
    "Setophaga_discolor",
    "Pipilo_erythrophthalmus",
    "Pica_nuttalli",
    "Toxostoma_longirostre",
    "Ardea_melanocephala",
    "Microcarbo_africanus",
    "Charadrius_wilsonia",
    "Spizella_breweri",
    "Patagioenas_flavirostris",
    "Chlidonias_hybrida",
    "Sterna_paradisaea",
    "Calidris_virgata",
    "Acridotheres_cristatellus",
    "Tigrisoma_lineatum",
    "Ciconia_nigra",
    "Oriolus_chinensis",
    "Chloroceryle_amazona",
    "Sylvia_curruca",
    "Pernis_ptilorhynchus",
    "Charadrius_alexandrinus",
    "Certhia_brachydactyla",
    "Myiodynastes_maculatus",
    "Larus_pacificus",
    "Treron_vernans",
    "Urocissa_erythroryncha",
    "Microcarbo_pygmeus",
    "Elseyornis_melanops",
    "Mycteria_leucocephala",
    "Gerygone_igata",
    "Fulica_cristata",
    "Turdus_leucomelas",
    "Brotogeris_jugularis",
    "Pycnonotus_aurigaster",
    "Lagonosticta_senegala",
    "Poecile_hudsonicus",
    "Ploceus_velatus",
    "Ploceus_cucullatus",
    "Plocepasser_mahali",
    "Calidris_ruficollis",
    "Tringa_stagnatilis",
    "Empidonax_virescens",
    "Polemaetus_bellicosus",
    "Lamprotornis_superbus",
    "Jynx_torquilla",
    "Buteo_rufinus",
    "Cecropis_daurica",
    "Anser_fabalis",
    "Coscoroba_coscoroba",
    "Anthus_petrosus",
    "Torgos_tracheliotos",
    "Lamprotornis_nitens",
    "Habia_fuscicauda",
    "Calonectris_diomedea",
    "Vanellus_senegallus",
    "Polioptila_californica",
    "Cinnyris_asiaticus",
    "Saltator_aurantiirostris",
    "Burhinus_vermiculatus",
    "Charadrius_collaris",
    "Cyanocorax_beecheii",
    "Myiozetetes_cayanensis",
    "Dicrurus_leucophaeus",
    "Phoeniculus_purpureus",
    "Colaptes_chrysoides",
    "Lophaetus_occipitalis",
    "Ixobrychus_minutus",
    "Ardea_goliath",
    "Stercorarius_longicaudus",
    "Aethopyga_siparaja",
    "Prinia_subflava",
    "Fluvicola_nengeta",
    "Lophoceros_alboterminatus",
    "Vireo_atricapilla",
    "Halcyon_senegalensis",
    "Stelgidopteryx_ruficollis",
    "Ortalis_cinereiceps",
    "Forpus_conspicillatus",
    "Trogon_rufus",
    "Anthracothorax_nigricollis",
    "Rhipidura_javanica",
    "Dendrocopos_syriacus",
    "Eucometis_penicillata",
    "Poecile_sclateri",
    "Emberiza_rustica",
    "Colaptes_punctigula",
    "Ocreatus_underwoodii",
    "Prionops_plumatus",
    "Antrostomus_vociferus",
    "Spheniscus_magellanicus"
]   
    print("The num of class:", len(classnames))
    return classnames