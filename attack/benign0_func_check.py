import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
from torchvision.datasets import CIFAR10, CIFAR100, STL10, ImageNet, ImageFolder
from datasets.dataset import ImageToHash_Attack
import numpy as np
import pandas as pd
import base64
from torchvision.transforms.functional import to_pil_image
import pdqhash
import cv2

from models.neuralhash import NeuralHash
from models.NeuralHash_track_is_true import NeuralHash_track_is_true
from utils.hashing import load_hash_matrix
from models.resnet_v5 import *
from utils.transforms import Rotate, Translate, ChangeSaturation, ChangeHue, ChangeContrast, ChangeBrightness, \
    JpegCompression, HorizontalFlipping, BlackBorder, CenterCrop, VerticalFlipping, RandomPerspectiveTransform, Grayscale, GaussianBlur, RandomErasing, ResizeTransform, PepperSaltNoise

def validate_target(value):
    valid_prefixes = ['photodna_nn_base', 'photodna_nn_adv', 'photodna_nn_cert', 'photodna', 'pdq', 'neuralhash_nn']
    # Check if the start of 'value' matches any of the valid prefixes
    if any(value.startswith(prefix) for prefix in valid_prefixes):
        return value
    else:
        raise argparse.ArgumentTypeError(f"{value} is an invalid value for --target")

def get_dataset(dataset_name: str, additional_transforms=None, target = 'photodna'):
    img_transform = get_coco_transforms(additional_transforms=additional_transforms, target=target)
    if dataset_name.lower() == 'coco_val':
        dataset = ImageToHash_Attack('../train_verify/data/coco-val.csv', '../train_verify/data', transform=img_transform)
    elif dataset_name.lower() == 'nsfw':
        dataset = ImageToHash_Attack('../train_verify/data/nsfw_val.csv', '../train_verify/data', transform=img_transform)
    elif dataset_name.lower() == 'cifar10_val':
        dataset = ImageToHash_Attack('../train_verify/data/cifar10_val.csv', '../train_verify/data', transform=img_transform)
    else:
        raise RuntimeError(f'Dataset with name {dataset_name} was not found.')

    return dataset


def get_coco_transforms(additional_transforms=None, target = 'photodna_nn'):
    if 'nn' in target:
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
    else:
        mean = torch.tensor([0, 0, 0])
        std = torch.tensor([1, 1, 1])
    transforms = [
        T.ConvertImageDtype(torch.float32),
        T.Resize((64,64)),
    ]
    if additional_transforms is not None and type(additional_transforms) == list:
        transforms.extend(additional_transforms)
    if 'nn' in target:
        img_transform = T.Compose(transforms + [T.Resize((64,64)), T.Normalize(mean=mean, std=std),])
    else:
        img_transform = T.Compose(transforms + [T.Normalize(mean=mean, std=std)])
    return img_transform

def get_neuralhash_transforms(additional_transforms=None):
    transforms = [
        # T.Resize((360, 360)),
        T.ConvertImageDtype(torch.float32),
        T.Resize((64, 64)),
    ]
    if additional_transforms is not None and type(additional_transforms) == list:
        transforms.extend(additional_transforms)
    # transforms.append(T.Lambda(lambda x: x * 2 - 1))
    img_transform = T.Compose(transforms)

    return img_transform

def get_translation_tuples(max_trans, trans_log_base, trans_steps):
    translations = []
    values = np.unique(
        np.ceil(
            np.logspace(0, np.log(max_trans) / np.log(trans_log_base), trans_steps, endpoint=True, base=trans_log_base)
        ).astype(int)
    )
    values = [0] + values.tolist()

    for hor_trans in values:
        for vert_trans in values:
            translations.append((hor_trans, vert_trans))

    return translations


def get_rotation_angles(max_rot_angle, rot_log_base, rot_steps):
    # create the list of angle and rotation values
    angles = np.unique(
        np.ceil(
            np.logspace(0, np.log(max_rot_angle) / np.log(rot_log_base), rot_steps, endpoint=True, base=rot_log_base)
        ).astype(int)
    )
    angles = np.flip(-angles).tolist() + [0] + angles.tolist()
    return angles

def get_rotation_angles_positive(max_rot_angle, rot_log_base, rot_steps):
    # create the list of angle and rotation values
    angles = np.unique(
        np.ceil(
            np.logspace(0, np.log(max_rot_angle) / np.log(rot_log_base), rot_steps, endpoint=True, base=rot_log_base)
        ).astype(int)
    )
    angles = [0] + angles.tolist()
    return angles


def get_neural_hashes(dataset, model, seed, device, batch_size=128, num_workers=8):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    binary_hashes = []
    hex_hashes = []
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc='Getting Neural Hashes', leave=False):
            x = x.to(device)
            hash = model(x).squeeze().unsqueeze(2)
            hash = torch.matmul(seed.repeat(len(x), 1, 1), hash)
            hash = torch.sign(hash).view(len(x), -1).cpu()
            # convert the tensor from [-1, 1] to [0, 1]
            hash = (hash > 0).type(torch.IntTensor)
            hash_bin = [''.join(list(map(str, x.tolist()))) for x in hash]
            hash_hex = ['{:0{}x}'.format(int(hash_bits, 2), len(hash_bits) // 4) for hash_bits in hash_bin]
            binary_hashes.extend(hash_bin)
            hex_hashes.extend(hash_hex)

    return binary_hashes, hex_hashes

def get_coco_hashes(dataset, model, device, batch_size=128, num_workers=8):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    numeric_hashes = []
    hex_hashes = []
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc='Getting COCO Hashes', leave=False):
            x = x.to(device)
            y = torch.relu(torch.round(model(x)))
            uint8_array = y.cpu().detach().numpy().astype(np.uint8)  ##hash bin with int
            hash_bin = [i for i in uint8_array]
            hash_hex = [base64.b64encode(i.tobytes()).decode("utf-8") for i in uint8_array]
            numeric_hashes.extend(hash_bin)
            hex_hashes.extend(hash_hex)
    return numeric_hashes, hex_hashes



def run_augmentation(dataset, model, device, augmentation, augmentation_inputs, file_paths, batch_size=64,
                     num_workers=8, target='photodna_nn', seed = None):
    for augm_input, file_path in tqdm(zip(augmentation_inputs, file_paths), desc=augmentation.__name__ if augmentation else 'Original', total=len(augmentation_inputs)):

        if augmentation is not None:
            if 'neuralhash' not in target:
                new_transforms = get_coco_transforms(additional_transforms=[augmentation(augm_input) if augm_input is not None else augmentation()], target= target)
                dataset.transforms = new_transforms
            else:
                new_transforms = get_neuralhash_transforms(additional_transforms=[augmentation(augm_input) if augm_input is not None else augmentation()])
                dataset.transforms = new_transforms

        if 'nn' in target:
            if os.path.exists(file_path):
                continue

            #make an empty dummy file to support multiple runs to work together at the same time
            if not os.path.exists(file_path):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                Path(file_path).touch(exist_ok=False)

            if 'neuralhash' not in target:
                binary_hashes, hex_hashes = get_coco_hashes(dataset, model, device, batch_size=batch_size,
                                                       num_workers=num_workers)
                hash_df = pd.DataFrame(columns=['image', 'hash_bin', 'hash_hex'])
                hash_df = hash_df.assign(hash_bin=binary_hashes, hash_hex=hex_hashes)
                if hasattr(dataset, 'imgs'):
                    hash_df = hash_df.assign(image=list(np.array(dataset.imgs)[:, 0]))

                if not os.path.exists(os.path.dirname(file_path)):
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                hash_df.to_csv(file_path)

            else:
                binary_hashes, hex_hashes = get_neural_hashes(dataset, model, seed, device, batch_size=batch_size,
                                                              num_workers=num_workers)
                hash_df = pd.DataFrame(columns=['image', 'hash_bin', 'hash_hex'])
                hash_df = hash_df.assign(hash_bin=binary_hashes, hash_hex=hex_hashes)
                if hasattr(dataset, 'imgs'):
                    hash_df = hash_df.assign(image=list(np.array(dataset.imgs)[:, 0]))

                if not os.path.exists(os.path.dirname(file_path)):
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                hash_df.to_csv(file_path)


        else:
            if 'pdq' in target:
                if os.path.exists(file_path):
                    continue
                # make an empty dummy file to support multiple runs to work together at the same time
                if not os.path.exists(file_path):
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    Path(file_path).touch(exist_ok=False)
                # data = []
                data = []
                for i in range(len(dataset)):
                    img_tensor, _ = dataset[i]  # get the transformed image tensor and its original name
                    img_np = img_tensor.numpy().transpose(1, 2, 0) * 255
                    img_np = img_np.astype(np.uint8)
                    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    hash_vector, _ = pdqhash.compute(img_cv2)
                    data.append({'image': i, 'hash_bin': hash_vector})
                hash_df = pd.DataFrame(data)
                if not os.path.exists(os.path.dirname(file_path)):
                    os.makedirs(os.path.dirname(file_path), exist_ok=False)
                hash_df.to_csv(file_path)

            else:
                if file_path.endswith('.csv'):
                    file_path = file_path[:-4]
                if os.path.exists(file_path):
                    continue
                if not os.path.exists(file_path):
                    os.makedirs(file_path, exist_ok=True)
                for i in range(len(dataset)):
                    img_tensor, _ = dataset[i]  # get the transformed image tensor and its original name
                    img = to_pil_image(img_tensor)  # Convert tensor to PIL Image for saving
                    file_name = f"{i}.jpg"
                    output_path = os.path.join(file_path, file_name)  # create the full output path
                    img.save(output_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='coco_val',
                        choices=['coco_val', 'nsfw', 'cifar10_val'], type=str,
                        help='The dataset that is used')
    parser.add_argument('--batch_size', default=128, type=int, help='The batch size used for inference')
    parser.add_argument('--max_rot_angle', default=64, type=int,
                        help='The angle (in degrees) by which the image is rotated clockwise and counterclockwise')
    parser.add_argument('--rot_log_base', default=2, type=int, help='The logarithm base')
    parser.add_argument('--rot_steps', default=7, type=int, help='The number of rotations steps')
    parser.add_argument('--max_trans', default=64, type=int,
                        help='The max translation in pixels by which the image is going to be translated')
    parser.add_argument('--trans_log_base', default=2, type=int, help='The logarithm base')
    parser.add_argument('--trans_steps', default=7, type=int,
                        help='The number of translation steps in vertical and horizontal direction, respectively')
    parser.add_argument('--device', default='cuda', type=str, help='The device used for inference')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='The number of workers that is used for loading the data')
    parser.add_argument('--output_dir', default='func_logs', type=str,
                        help='The output directory where the results are going to be saved as CSV files')
    parser.add_argument('--target', type=validate_target, default='photodna_nn_cert',
                        help='_nn means using neural network to train a model, the choices can be changes as your target')
    parser.add_argument('--model', default='../train_verify/saved_models/coco_photodna_ep1/ckpt_best.pth', type=str,
                        help='your model path if "nn", not applicable for neuralhash, since the path for neuralhash is hardcoded as "./models/model.pth"')


    args = parser.parse_args()

    if 'nn' in args.target:
        device = torch.device(args.device)
        if 'neuralhash' not in args.target:
            model = resnet_v5(num_classes=144, bn=True, input_dim=64)
            # model.load_state_dict(torch.load('../Normal-Training/64-coco-hash-resnetv5-l1-aug2-new.pt'))
            model.load_state_dict(torch.load(args.model))
            model = model.to(device)
            seed = None
        else:
            model = NeuralHash_track_is_true()
            model.load_state_dict(torch.load('./models/model.pth'))
            model = model.to(device)
            seed = torch.tensor(load_hash_matrix())
            seed = seed.to(device)

    else:
        model = None
        device = None
        seed = None

    output_dir = os.path.join(args.output_dir, f'{args.dataset}_{args.target}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = get_dataset(args.dataset, args.target)

    # get the rotation angles and the translation tuples
    angles = get_rotation_angles(args.max_rot_angle, args.rot_log_base, args.rot_steps)
    hue_values = list(range(-180, 180, 30))
    saturation_values = list(np.linspace(0, 2, 9, endpoint=True))
    brightness_values = list(np.linspace(0, 2, 9, endpoint=True))
    contrast_values = list(np.linspace(0, 2, 9, endpoint=True))
    compression_values = [100] + list(
        (100 - np.ceil(np.logspace(0, np.log(100) / np.log(1.5), 10, endpoint=True, base=1.5))).clip(0, 100)
    )

    crop_values = list(
        filter(
            lambda x: x != 63,  # Change this value if you specifically want to exclude a different size
            [64] + list(64 - np.append(np.logspace(0, 5, 6, base=2, endpoint=True, dtype=int), [32]))
        )
    )
    downsizing_values = list(
        filter(
            lambda x: x != 63,  # Ensure consistency in what's being excluded
            [64] + list(64 - np.append(np.logspace(0, 5, 6, base=2, endpoint=True, dtype=int), [32]))
        )
    )

    perceptive_values = list([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])
    gaussianBlur_values = list([0.0000000001, 1, 2, 3, 4, 5, 6, 8, 10])
    erasing_values = list(np.linspace(0, 0.3, 6, endpoint=True))
    gray_values = list(np.linspace(0, 1, 9, endpoint=True))
    resize_values = list([100, 80, 64, 56, 48, 32])
    pepper_values = list([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])


    # iterations = len(angles) + len(translations) + len(hue_values) + len(saturation_values) + \
    #              len(brightness_values) + len(contrast_values) + len(compression_values) + len(crop_values) + len(downsizing_values) + 1

    # get the initial hashes
    run_augmentation(
        dataset,
        model,
        device,
        None,
        [None],
        [os.path.join(output_dir, f'{args.dataset}_original.csv')],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target = args.target,
        seed = seed
    )

    # test the robustness against rotations
    run_augmentation(
        dataset,
        model,
        device,
        Rotate,
        angles,
        [os.path.join(output_dir, 'rotation', f'{args.dataset}_rotation_{angle}.csv') for angle in angles],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target=args.target,
        seed = seed
    )



    # test the robustness against hue changes
    run_augmentation(
        dataset,
        model,
        device,
        ChangeHue,
        hue_values,
        [os.path.join(output_dir, 'hue', f'{args.dataset}_hue_{hue}.csv') for hue in hue_values],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target=args.target,
        seed = seed
    )

    #test the robustness against saturation changes
    run_augmentation(
        dataset,
        model,
        device,
        ChangeSaturation,
        saturation_values,
        [os.path.join(output_dir, 'saturation', f'{args.dataset}_saturation_{saturation}.csv') for saturation in saturation_values],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target=args.target,
        seed = seed
    )

    # test the robustness against brightness changes
    run_augmentation(
        dataset,
        model,
        device,
        ChangeBrightness,
        brightness_values,
        [os.path.join(output_dir, 'brightness', f'{args.dataset}_brightness_{brightness}.csv') for brightness in brightness_values],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target=args.target,
        seed = seed
    )

    # test the robustness against contrast changes
    run_augmentation(
        dataset,
        model,
        device,
        ChangeContrast,
        contrast_values,
        [os.path.join(output_dir, 'contrast', f'{args.dataset}_contrast_{contrast}.csv') for contrast in contrast_values],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target=args.target,
        seed = seed
    )

    #test the robustness against compression
    run_augmentation(
        dataset,
        model,
        device,
        JpegCompression,
        compression_values,
        [os.path.join(output_dir, 'compression', f'{args.dataset}_compression_{compression}.csv') for compression in compression_values],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target=args.target,
        seed = seed
    )

    run_augmentation(
        dataset,
        model,
        device,
        CenterCrop,
        crop_values,
        [os.path.join(output_dir, 'crop', f'{args.dataset}_crop_{crop}.csv') for crop in crop_values],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target=args.target,
        seed = seed
    )

    run_augmentation(
        dataset,
        model,
        device,
        HorizontalFlipping,
        [None],
        [os.path.join(output_dir, 'hflip', f'{args.dataset}_hflip.csv')],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target=args.target,
        seed = seed
    )

    run_augmentation(
        dataset,
        model,
        device,
        VerticalFlipping,
        [None],
        [os.path.join(output_dir, 'vflip', f'{args.dataset}_vflip.csv')],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target=args.target,
        seed = seed
    )

    run_augmentation(
        dataset,
        model,
        device,
        BlackBorder,
        downsizing_values,
        [os.path.join(output_dir, 'downsizing', f'{args.dataset}_downsizing_{size}.csv') for size in downsizing_values],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target=args.target,
        seed = seed
    )

    run_augmentation(
        dataset,
        model,
        device,
        ResizeTransform,
        resize_values,
        [os.path.join(output_dir, 'resizing', f'{args.dataset}_resizing_{size}.csv') for size in resize_values],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target=args.target,
        seed = seed
    )

    run_augmentation(
        dataset,
        model,
        device,
        RandomPerspectiveTransform,
        perceptive_values,
        [os.path.join(output_dir, 'perceptive', f'{args.dataset}_perceptive_{size}.csv') for size in perceptive_values],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target=args.target,
        seed = seed
    )



    run_augmentation(
        dataset,
        model,
        device,
        GaussianBlur,
        gaussianBlur_values,
        [os.path.join(output_dir, 'gaussianBlur', f'{args.dataset}_gaussianBlur_{0 if size < 0.001 else size}.csv') for
         size in gaussianBlur_values],
    batch_size=args.batch_size,
        num_workers=args.num_workers,
        target=args.target,
        seed = seed
    )

    run_augmentation(
        dataset,
        model,
        device,
        RandomErasing,
        [(i, i) for i in erasing_values],
        [os.path.join(output_dir, 'erasing', f'{args.dataset}_erasing_{size}.csv') for size in erasing_values],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target=args.target,
        seed = seed
    )

    run_augmentation(
        dataset,
        model,
        device,
        Grayscale,
        gray_values,
        [os.path.join(output_dir, 'gray', f'{args.dataset}_gray_{size}.csv') for size in gray_values],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target=args.target,
        seed = seed
    )


    run_augmentation(
        dataset,
        model,
        device,
        PepperSaltNoise,
        pepper_values,
        [os.path.join(output_dir, 'pepperSalt', f'{args.dataset}_pepperSalt_{size}.csv') for size in pepper_values],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target=args.target,
        seed=seed
    )






if __name__ == '__main__':
    main()
