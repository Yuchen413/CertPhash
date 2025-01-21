import multiprocessing
import torch
from torch.utils import data
from functools import partial
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset

from torchvision.io import read_image, ImageReadMode
import numpy as np
import base64
import csv
import os
import random

import torchvision.transforms.functional as TF

class RotateByAngle:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        return TF.rotate(img, self.angle)


class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

class ImageToHash(Dataset):
    def __init__(self, hashes_csv, image_dir, resize=None):
        self.image_dir = image_dir
        self.resize = resize
        self.names_and_hashes = []
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        with open(hashes_csv) as f:
            r = csv.reader(f)
            for line in r:
                path = line[0]
                h = np.array(list(base64.b64decode(line[1])), dtype=np.uint8)
                self.names_and_hashes.append((path, h))

        self.transforms = transforms.Compose([
            transforms.Resize(self.resize) if self.resize is not None else transforms.Lambda(lambda x: x),
            transforms.ConvertImageDtype(torch.float32),  # Converts to float and scales to [0, 1]
            transforms.Normalize(mean=self.mean, std=self.std)  # Normalizes the image
        ])

    def __len__(self):
        return len(self.names_and_hashes)

    def __getitem__(self, idx):
        name, h = self.names_and_hashes[idx]
        img_path = os.path.join(self.image_dir, name)
        img = read_image(img_path, mode=ImageReadMode.RGB)

        img = self.transforms(img)

        return img.float(), torch.tensor(h).float()

# class ImageToHashAugmented(Dataset):
#     def __init__(self, hashes_csv, image_dir, resize, num_augmented=0):
#         self.image_dir = image_dir
#         self.resize = resize
#         self.num_augmented = num_augmented
#         self.names_and_hashes = []
#         self.mean = torch.tensor([0.485, 0.456, 0.406])
#         self.std = torch.tensor([0.229, 0.224, 0.225])
#         with open(hashes_csv) as f:
#             r = csv.reader(f)
#             for line in r:
#                 path = line[0]
#                 h = np.array(list(base64.b64decode(line[1])), dtype=np.uint8)
#                 # h = np.unpackbits(np.frombuffer(base64.b64decode(line[1]), dtype=np.uint8)) #into 01010110
#                 self.names_and_hashes.append((path, h))
#
#         self.existing_transforms =[transforms.Resize(self.resize) if self.resize is not None else transforms.Lambda(lambda x: x),]
#
#         # Define a list of possible transformations
#         self.possible_transforms = [
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomVerticalFlip(),
#             transforms.RandomRotation(64),
#             transforms.ColorJitter(brightness=(0,2), contrast=(0,2), saturation=(0,2), hue=0.5),
#             transforms.RandomPerspective(distortion_scale=0.2, p=1),
#             transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2)),
#             transforms.RandomCrop(64, 2, padding_mode='edge'),
#             # Add more transformations here
#         ]
#
#     def __len__(self):
#         return len(self.names_and_hashes)
#
#     def __getitem__(self, idx):
#         name, h = self.names_and_hashes[idx]
#         img_path = os.path.join(self.image_dir, name)
#         img = read_image(img_path, mode=ImageReadMode.RGB)
#
#         # Apply base transforms (resize, normalization)
#
#         # Apply random augmentations, could be 0, which is no augment
#         if self.num_augmented > 0:
#             num_transformations_to_apply = random.randint(0, self.num_augmented)
#             if num_transformations_to_apply > 0:
#                 augmented_transforms = random.sample(self.possible_transforms, num_transformations_to_apply)
#                 # self.transforms = transforms.Compose(augmented_transforms)
#             else: augmented_transforms = []
#         else: augmented_transforms = []
#
#         self.transforms = transforms.Compose(self.existing_transforms + augmented_transforms +
#                                              [transforms.ConvertImageDtype(torch.float32),
#                                               transforms.Normalize(mean=self.mean, std=self.std)])
#
#         img = self.transforms(img)
#
#         return img, torch.tensor(h).float()

class ImageToHashAugmented(Dataset):
    def __init__(self, hashes_csv, image_dir, resize, num_augmented=0):
        self.image_dir = image_dir
        self.resize = resize
        self.num_augmented = num_augmented
        self.names_and_hashes = []
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])

        with open(hashes_csv) as f:
            r = csv.reader(f)
            for line in r:
                path = line[0]
                h = np.array(list(base64.b64decode(line[1])), dtype=np.uint8)
                # h = np.unpackbits(np.frombuffer(base64.b64decode(line[1]), dtype=np.uint8)) #into 01010110
                self.names_and_hashes.append((path, h))

        # self.rotate_transforms = [
        #     transforms.RandomRotation(64),
        #     transforms.RandomRotation(16),
        #     ]

        # self.crop_transforms = [
        #     transforms.RandomResizedCrop(size=64, scale=(0.5, 1.0))]

        # Define a list of possible transformations
        er_scale = random.uniform(0, 0.5)
        self.base_transforms = [
            transforms.RandomRotation(64),
            transforms.RandomRotation(16),
            transforms.RandomResizedCrop(size=64, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=(0,2), contrast=(0,2), saturation=(0,2), hue=0.5),
            transforms.RandomPerspective(distortion_scale=random.uniform(0.05, 0.5), p=1),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.RandomErasing(p=1, scale=(er_scale, er_scale), value=0)
        ]


    def __len__(self):
        return len(self.names_and_hashes)

    def __getitem__(self, idx):
        name, h = self.names_and_hashes[idx]
        img_path = os.path.join(self.image_dir, name)
        img = read_image(img_path, mode=ImageReadMode.RGB).float() / 255.0

        transformations = []
        if self.num_augmented > 0:
            num_transformations_to_apply = random.randint(0, self.num_augmented)
            if num_transformations_to_apply > 0:
                transformations += random.sample(self.base_transforms,min(len(self.base_transforms), self.num_augmented))
                # transformations += self.crop_transforms
                # transformations.append(random.choice(self.rotate_transforms))
                # transformations.append(random.choice(self.crop_transforms))

        transformations += [
            transforms.Resize((self.resize, self.resize)),
            transforms.Normalize(mean=self.mean, std=self.std)
        ]

        transform_pipeline = transforms.Compose(transformations)
        img = transform_pipeline(img)

        return img, torch.tensor(h).float()

class ImageToHashAugmented_PDQ_with_class(Dataset):
    def __init__(self, hashes_csv, image_dir, resize, num_augmented=0):
        self.image_dir = image_dir
        self.resize = resize
        self.num_augmented = num_augmented
        self.names_and_hashes_and_clasees = []
        with open(hashes_csv) as f:
            r = csv.reader(f)
            for line in r:
                path = line[0]
                c =torch.tensor(int((path.split('/')[-1].split('_')[0])),dtype=torch.float)
                # h = np.array(list(base64.b64decode(line[1])), dtype=np.uint8)
                h = torch.tensor([int(bit) for bit in line[1].strip('[]').split()], dtype=torch.float)
                # h = np.unpackbits(np.frombuffer(base64.b64decode(line[1]), dtype=np.uint8)) #into 01010110
                self.names_and_hashes_and_clasees.append((path, h, c))

        # self.rotate_transforms = [
        #     transforms.RandomRotation(64),
        #     transforms.RandomRotation(16),
        #     ]
        #
        # self.crop_transforms = [
        #     # transforms.RandomResizedCrop(size=64, scale=(0.5, 1.0)),]

        # Define a list of possible transformations
        er_scale = random.uniform(0, 0.5)
        self.base_transforms = [
            transforms.RandomRotation(64),
            transforms.RandomRotation(16),
            transforms.RandomResizedCrop(size=64, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=(0,2), contrast=(0,2), saturation=(0,2), hue=0.5),
            transforms.RandomPerspective(distortion_scale=random.uniform(0.05, 0.5), p=1),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.RandomErasing(p=1, scale=(er_scale, er_scale), value=0)
        ]

    def __len__(self):
        return len(self.names_and_hashes_and_clasees)

    def __getitem__(self, idx):
        name, h, c = self.names_and_hashes_and_clasees[idx]
        img_path = os.path.join(self.image_dir, name)
        img = read_image(img_path).float() / 255.0

        transformations = []
        if self.num_augmented > 0:
            num_transformations_to_apply = random.randint(0, self.num_augmented)
            if num_transformations_to_apply > 0:
                transformations += random.sample(self.base_transforms,min(len(self.base_transforms), self.num_augmented))
                # transformations += self.crop_transforms
                # transformations.append(random.choice(self.rotate_transforms))
                # transformations.append(random.choice(self.crop_transforms))

        transformations += [
            transforms.Resize(self.resize),
        ]

        transform_pipeline = transforms.Compose(transformations)
        img = transform_pipeline(img)

        return img, h, c

class ImageToHashAugmented_PDQ(Dataset):
    def __init__(self, hashes_csv, image_dir, resize, num_augmented=0):
        self.image_dir = image_dir
        self.resize = resize
        self.num_augmented = num_augmented
        self.names_and_hashes = []
        with open(hashes_csv) as f:
            r = csv.reader(f)
            for line in r:
                path = line[0]
                # h = np.array(list(base64.b64decode(line[1])), dtype=np.uint8)
                h = torch.tensor([int(bit) for bit in line[1].strip('[]').split()], dtype=torch.float)
                # h = np.unpackbits(np.frombuffer(base64.b64decode(line[1]), dtype=np.uint8)) #into 01010110
                self.names_and_hashes.append((path, h))

        # self.rotate_transforms = [
        #     transforms.RandomRotation(64),
        #     transforms.RandomRotation(16),
        #     ]
        #
        # self.crop_transforms = [
        #     transforms.RandomCrop(self.resize, 2, padding_mode='edge'),]

        er_scale = random.uniform(0, 0.5)
        self.base_transforms = [
            transforms.RandomRotation(64),
            transforms.RandomRotation(16),
            transforms.RandomResizedCrop(size=64, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=(0,2), contrast=(0,2), saturation=(0,2), hue=0.5),
            transforms.RandomPerspective(distortion_scale=random.uniform(0.05, 0.5), p=1),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.RandomErasing(p=1, scale=(er_scale, er_scale), value=0)
        ]

    def __len__(self):
        return len(self.names_and_hashes)

    def __getitem__(self, idx):
        name, h = self.names_and_hashes[idx]
        img_path = os.path.join(self.image_dir, name)
        img = read_image(img_path).float() / 255.0

        transformations = []
        if self.num_augmented > 0:
            num_transformations_to_apply = random.randint(0, self.num_augmented)
            if num_transformations_to_apply > 0:
                transformations += random.sample(self.base_transforms,min(len(self.base_transforms), self.num_augmented))
                # transformations += self.crop_transforms
                # transformations.append(random.choice(self.rotate_transforms))
                # transformations.append(random.choice(self.crop_transforms))

        transformations += [
            transforms.Resize(self.resize),
        ]

        transform_pipeline = transforms.Compose(transformations)
        img = transform_pipeline(img)

        return img, h


def load_data(args, data, batch_size, test_batch_size, use_index=False, aug=True):
    if 'normal' in data:
        num_aug = 0
    else:
        num_aug = 2

    if 'MNIST' in data:
        input_dim = 28
        dummy_input = torch.randn(2, 1, 28, 28)
        mean = torch.tensor([0.0])
        std = torch.tensor([1.0])
        train_data = ImageToHashAugmented_PDQ('data/mnist/mnist_train.csv', 'data', resize=input_dim, num_augmented=num_aug)
        test_data = ImageToHashAugmented_PDQ('data/mnist/mnist_test.csv', 'data', resize=input_dim, num_augmented=0)

    elif 'coco' in data:
        input_dim = 64
        dummy_input = torch.randn(2, 3, input_dim, input_dim)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        train_data = ImageToHashAugmented('data/coco-train.csv', 'data', resize=input_dim, num_augmented=num_aug)
        test_data = ImageToHashAugmented('data/coco-val.csv', 'data', resize=input_dim, num_augmented=0)

    elif 'nsfw' in data:
        input_dim = 64
        dummy_input = torch.randn(2, 3, input_dim, input_dim)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        train_data = ImageToHashAugmented('data/nsfw_train.csv', 'data', resize=input_dim, num_augmented=num_aug)
        test_data = ImageToHashAugmented('data/nsfw_val.csv', 'data', resize=input_dim, num_augmented=0)

    train_data = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    test_data = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, pin_memory=True, num_workers=0)
        
    train_data.mean = test_data.mean = mean
    train_data.std = test_data.std = std  

    for loader in [train_data, test_data]:
        loader.mean, loader.std = mean, std
        loader.data_max = data_max = torch.reshape((1. - mean) / std, (1, -1, 1, 1))
        loader.data_min = torch.reshape((0. - mean) / std, (1, -1, 1, 1))

    dummy_input = dummy_input.to(args.device)

    return dummy_input, train_data, test_data
