import os
import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode

# def load_and_preprocess_img(img_path, device, resize=True, normalize=True):
#     image = Image.open(img_path).convert('RGB')
#     if resize:
#         image = image.resize([360, 360])
#     arr = np.array(image).astype(np.float32) / 255.0
#     if normalize:
#         arr = arr * 2.0 - 1.0
#     arr = arr.transpose(2, 0, 1)
#     tensor = torch.tensor(arr).unsqueeze(0)
#     tensor = tensor.to(device)
#     return tensor

def normalize(img,data):
    if data == 'mnist':
        mean = torch.tensor([0])
        std = torch.tensor([1])
    else:
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
    normal = transforms.Normalize(mean=mean, std=std)
    return normal(img)


def denormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    denormal = transforms.Normalize((-1 * mean / std), (1.0 / std))
    return denormal(img)

def load_and_preprocess_img(img_path, device, data = 'mnist'):
    if data == 'mnist':
        transform = transforms.Compose([
                transforms.ConvertImageDtype(torch.float32),  # Converts to float and scales to [0, 1]
            ])
        image = read_image(img_path, mode=ImageReadMode.GRAY)
    else:
        transform = transforms.Compose([
                transforms.Resize((64,64)),
                transforms.ConvertImageDtype(torch.float32),  # Converts to float and scales to [0, 1]
            ])
        image = read_image(img_path, mode=ImageReadMode.RGB)
    tensor= transform(image)
    tensor = tensor.unsqueeze(0).to(device)
    return tensor

def save_images(img: torch.tensor, folder, filename):
    # mean = torch.tensor([0.485, 0.456, 0.406])
    # std = torch.tensor([0.229, 0.224, 0.225])
    # img = denormalize(img)
    img = img.detach().cpu()
    img = img.clamp(0.0, 1.0)
    img = img.squeeze()
    path = os.path.join(folder, f'{filename}.png')
    save_image(img, path)
