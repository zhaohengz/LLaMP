#external libs
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFilter
import os
import random
from os.path import join as ospj
from glob import glob 
#torch libs
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
#local libs
from utils.utils import get_norm_values, chunks
from itertools import chain, product
import json
from data.coop import *
from argparse import Namespace
from torchvision.transforms.functional import InterpolationMode

from data.randaugment import RandomAugment

device = 'cuda' if torch.cuda.is_available() else 'cpu'

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

class ImageLoader:
    def __init__(self, root):
        self.root_dir = root

    def __call__(self, img):
        img = Image.open(ospj(self.root_dir,img)).convert('RGB') #We don't want alpha
        return img
    
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x    


def dataset_transform(phase, norm_family ='clip', rand_aug=False):
    '''
        Inputs
            phase: String controlling which set of transforms to use
            norm_family: String controlling which normaliztion values to use
        
        Returns
            transform: A list of pytorch transforms
    '''
    mean, std = get_norm_values(norm_family=norm_family)

    if phase == 'train':
        if rand_aug:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                # transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    elif phase == 'val' or phase == 'test':
        transform = transforms.Compose([
            # transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif phase == 'all':
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise ValueError('Invalid transform')

    return transform

def filter_data(all_data, pairs_gt, topk = 5):
    '''
    Helper function to clean data
    '''
    valid_files = []
    with open('/home/ubuntu/workspace/top'+str(topk)+'.txt') as f:
        for line in f:
            valid_files.append(line.strip())

    data, pairs, attr, obj  = [], [], [], []
    for current in all_data:
        if current[0] in valid_files:
            data.append(current)
            pairs.append((current[1],current[2]))
            attr.append(current[1])
            obj.append(current[2])
            
    counter = 0
    for current in pairs_gt:
        if current in pairs:
            counter+=1
    print('Matches ', counter, ' out of ', len(pairs_gt))
    print('Samples ', len(data), ' out of ', len(all_data))
    return data, sorted(list(set(pairs))), sorted(list(set(attr))), sorted(list(set(obj)))

# Dataset class now

DATASET_CLASSMAP = {
    'Caltech101': Caltech101,
    'FGVCAircraft': FGVCAircraft,
    'EuroSAT': EuroSAT,
    'ImageNet': ImageNet,
    'StanfordCars': StanfordCars,
    'DescribableTextures': DTD,
    'Food101': Food101,
    'OxfordPets': OxfordPets,
    'OxfordFlowers': OxfordFlowers,
    'SUN397': SUN397,
    'UCF101': UCF101,
    'ImageNetSketch': ImageNetSketch,
    'ImageNetV2': ImageNetV2,
    'ImageNetA': ImageNetA,
    'ImageNetR': ImageNetR,
}

class MetaDataset(Dataset):
    '''
    Inputs
        root: String of base dir of dataset
        phase: String train, val, test
        split: String dataset split
        subset: Boolean if true uses a subset of train at each epoch
        num_negs: Int, numbers of negative pairs per batch
        pair_dropout: Percentage of pairs to leave in current epoch
    '''
    def __init__(
        self,
        phase,
        dataset=None,
        seed=1,
        return_images=False,
        num_shots=16,
        num_template=1,
        rand_aug=False,
        few_shot=False
    ):
        self.phase = phase
        self.return_images = return_images

        dataset_args = Namespace(
            SEED=seed,
            NUM_SHOTS=num_shots,
            SUBSAMPLE_CLASSES='new' if phase == 'test' else 'base',
        )

        if few_shot:
            dataset_args.SUBSAMPLE_CLASSES = 'all'

        self.dataset = DATASET_CLASSMAP[dataset](dataset_args)
        self.template = CUSTOM_TEMPLATES[dataset]

        self.classnames = self.dataset.classnames
        self.idx2label = self.dataset.lab2cname
        self.loader = ImageLoader('')
        self.transform = dataset_transform(self.phase, 'clip', rand_aug=rand_aug)
        print(self.transform)
        self.num_template = num_template

        self.data_dir = self.dataset.dataset_dir

        if phase == 'train':
            self.dataset = self.dataset.train_x
        else:
            self.dataset = self.dataset.test


    def __getitem__(self, index):
        '''
        Call for getting samples
        '''

        data_sample = self.dataset[index]

        # Decide what to output
        pil_img = self.loader(data_sample.impath)
        img_1 = self.transform(pil_img)
        # img_2 = self.transform(pil_img)

        data = [img_1, img_1, data_sample.label]

        if self.return_images:
            data.append(data_sample.path)

        return data
    
    def __len__(self):
        '''
        Call for length
        '''
        return len(self.dataset)

if __name__ == '__main__':
    from argparse import Namespace
    from flags import DATA_FOLDER

    args = Namespace(
        dataset='FGVCAircraft',
        train_only=True,
        num_shots=16,
        seed=1
    )

    dset = DatasetCoOp(
        phase='train',
        dataset=args.dataset,
        num_shots=args.num_shots,
        seed=args.seed,
    )

    dataloader = torch.utils.data.DataLoader(
        dset,
        batch_size=3,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    for d in dataloader:
        pass
