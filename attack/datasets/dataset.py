from torch.utils.data import Dataset
import torchvision
from torchvision.io import read_image, ImageReadMode
import numpy as np
import torch
from torchvision import transforms
import random

import base64
import csv
import os


class HashToImage(Dataset):
    def __init__(self, hashes_csv, image_dir):
        self.image_dir = image_dir
        self.names_and_hashes = []
        with open(hashes_csv) as f:
            r = csv.reader(f)
            for line in r:
                path = line[0]
                h = np.array(list(base64.b64decode(line[1])), dtype=np.uint8)
                self.names_and_hashes.append((path, h))

    def __len__(self):
        return len(self.names_and_hashes)

    def __getitem__(self, idx):
        name, h = self.names_and_hashes[idx]
        img_path = os.path.join(self.image_dir, name)
        img = read_image(img_path, mode=ImageReadMode.RGB)
        return torch.tensor(h), img


# class ImageToHash(Dataset):
#     def __init__(self, hashes_csv, image_dir):
#         self.image_dir = image_dir
#         self.names_and_hashes = []
#         with open(hashes_csv) as f:
#             r = csv.reader(f)
#             for line in r:
#                 path = line[0]
#                 h = np.array(list(base64.b64decode(line[1])), dtype=np.uint8)
#                 self.names_and_hashes.append((path, h))
#
#     def __len__(self):
#         return len(self.names_and_hashes)
#
#     def __getitem__(self, idx):
#         name, h = self.names_and_hashes[idx]
#         img_path = os.path.join(self.image_dir, name)
#         img = read_image(img_path, mode=ImageReadMode.RGB)
#         return img, torch.tensor(h)


class ImageToHash_Attack(Dataset):
    def __init__(self, hashes_csv, image_dir, transform):
        self.image_dir = image_dir
        self.transforms = transform
        self.names_and_hashes = []
        with open(hashes_csv) as f:
            r = csv.reader(f)
            for line in r:
                path = line[0]
                h = np.array(list(base64.b64decode(line[1])), dtype=np.uint8)
                self.names_and_hashes.append((path, h))

    def __len__(self):
        return len(self.names_and_hashes)

    def __getitem__(self, idx):
        name, h = self.names_and_hashes[idx]
        img_path = os.path.join(self.image_dir, name)
        img = read_image(img_path, mode=ImageReadMode.RGB)
        img = self.transforms(img)
        return img, torch.tensor(h).float()



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

        self.transforms = transforms.Compose([
            transforms.Resize(self.resize) if self.resize is not None else transforms.Lambda(lambda x: x),
            transforms.ConvertImageDtype(torch.float32),  # Converts to float and scales to [0, 1]
            transforms.Normalize(mean=self.mean, std=self.std)  # Normalizes the image
        ])

        # Define a list of possible transformations
        self.possible_transforms = [
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=1),
            transforms.RandomPerspective(distortion_scale=0.2, p=1),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.RandomResizedCrop(size=(100, 100), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            # Add more transformations here
        ]

    def __len__(self):
        return len(self.names_and_hashes) * (1 + self.num_augmented)

    def __getitem__(self, idx):
        original_idx = idx // (1 + self.num_augmented)
        name, h = self.names_and_hashes[original_idx]
        img_path = os.path.join(self.image_dir, name)
        img = read_image(img_path, mode=ImageReadMode.RGB)

        # Determine the type of image (original or which augmented)
        mod_idx = idx % (1 + self.num_augmented)

        # Apply specific transformations or return original
        if mod_idx == 0:
            # Return the original image, no augmentation
            pass
        else:
            selected_transforms = random.sample(self.possible_transforms, self.num_augmented)
            transform = selected_transforms[mod_idx - 1]  # Use mod_idx-1 to map to 0-indexed list
            img = transform(img)

        img = self.transforms(img)


        return img, torch.tensor(h).float(), original_idx  # Return the original image index


