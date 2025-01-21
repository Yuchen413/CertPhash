import argparse
import concurrent.futures
import csv
import os
import threading
import warnings
warnings.filterwarnings("ignore")
from os.path import isfile, join
from random import randint

import torch
import torchvision.transforms as T
from skimage import feature

from models.resnet_v5 import resnet_v5
from models.resnet import resnet
import shutil

# from losses.hinge_loss import hinge_loss, hinge_loss_coco
# from losses.mse_loss import mse_loss_coco
# from losses.quality_losses import ssim_loss
# from losses.customized_loss import l_infinity_loss
# from utils.hashing import compute_hash_coco
from utils.image_processing import load_and_preprocess_img,save_images, normalize, denormalize
from utils.logger import Logger
import threading
from tqdm import tqdm
import numpy as np
import random
import pandas as pd


def optimization_thread(url_list, device, logger, args, model_path, epsilon, data, target_hashes=None,  bin_hex_hash_dict=None , target_hash_dict=None):
    print(f'Process {threading.get_ident()} started')
    if data == 'mnist':
        model = resnet(in_ch=1, in_dim=28)
        theshold = 90
    else:
        model = resnet_v5(input_dim=64)
        theshold = 1800
    model_weights = torch.load(model_path)
    model.load_state_dict(model_weights)
    model.eval()
    model.cuda()
    url_list_copy = url_list.copy()

    # Start optimizing images
    while (url_list != []):

        print(len(url_list))
        img = url_list.pop(0)  # Pop the first image from the original list
        if data == 'nsfw':
            target_hashes = target_hashes
        else:
            target_hashes = []
            if data == 'mnist':
                # Since the same class in mnist looks very similar, so they should be not considered as collision
                this_round = [x for x in url_list_copy if
                              x.split('/')[-1].split('_')[0] != img.split('/')[-1].split('_')[0]]
            else:
                this_round = [x for x in url_list_copy if x != img]
            for random_img in this_round:
                tensor = load_and_preprocess_img(random_img, device, data)
                with torch.no_grad():
                    target_hash = torch.relu(torch.round(model(normalize(tensor,data))))
                target_hashes.append(target_hash.squeeze(0))
            target_hashes = torch.stack(target_hashes)


        # Store and reload source image to avoid image changes due to different formats
        #todo load_img Change to per data
        source = load_and_preprocess_img(img, device,data)
        input_file_name = img.rsplit(sep='/', maxsplit=1)[1].split('.')[0]
        if args.output_folder != '':
            save_images(source, args.output_folder, f'{input_file_name}')
        source_orig = source.clone()
        delta = torch.zeros_like(source, requires_grad=True)

        with torch.no_grad():
            outputs_unmodified = model(normalize(source,data))
            unmodified_hash_bin = torch.relu(torch.round(outputs_unmodified))
            l1_loss = torch.nn.L1Loss(reduction='sum')
            l1_loss_mean = torch.nn.L1Loss(reduction='mean')
            l1_loss_none = torch.nn.L1Loss(reduction='none')
            l2_loss = torch.nn.MSELoss()

            loss = l1_loss_none(unmodified_hash_bin,target_hashes).sum(dim=-1)

            if data == 'nsfw':
                _, idx = torch.min(loss, dim=0)
                target_hash = target_hashes[idx.item()]
                target_hash_str = str(target_hash.cpu().tolist())
                target_hash_hex = bin_hex_hash_dict[target_hash_str]
                target_hash = target_hash.unsqueeze(0).float()

            else:
                value, idx = torch.min(loss, dim=0)
                target_hash = target_hashes[idx.item()]
                target_hash = target_hash.unsqueeze(0)
                target_path = this_round[idx.item()]


            # print('This is the path of preimage:', target_path)

            if args.edges_only:
                # Compute edge mask
                transform = T.Compose(
                    [T.ToPILImage(), T.Grayscale(), T.ToTensor()])
                image_gray = transform(source.squeeze()).squeeze()
                image_gray = image_gray.cpu().numpy()
                edges = feature.canny(image_gray, sigma=3).astype(int)
                edge_mask = torch.from_numpy(edges).to(device)

        #Apply attack
        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                params=[delta], lr=args.learning_rate)
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(
                params=[delta], lr=args.learning_rate)
        else:
            raise RuntimeError(
                f'{args.optimizer} is no valid optimizer class. Please select --optimizer out of [Adam, SGD]')


        for i in tqdm(range(1000)):
            outputs_source = model(normalize(source+delta,data))
            target_loss = l2_loss(outputs_source, target_hash)
            total_loss = target_loss
            # visual_loss = -1 * args.ssim_weight * \
            #               ssim_loss(source_orig, source+delta)
            # total_loss = target_loss + visual_loss


            optimizer.zero_grad()
            total_loss.backward()
            if args.edges_only:
                optimizer.param_groups[0]['params'][0].grad *= edge_mask

            optimizer.step()

            with torch.no_grad():
                delta.clamp_(-epsilon, epsilon) #l_inf norm
                # norm_delta = torch.norm(delta, p=2)  # Calculate the L2 norm of delta
                # if norm_delta > epsilon:
                #     # Scale delta to have an L2 norm of epsilon
                #     delta = delta * (epsilon / norm_delta)

                # Check for hash changes
                if i % args.check_interval == 0:
                    with torch.no_grad():
                        current_img = source + delta
                        check_output = model(normalize(current_img,data))
                        source_hash_hex = torch.relu(torch.round(check_output)).int()
                        if l1_loss(source_hash_hex, target_hash) < theshold:
                            # Compute metrics in the [0, 1] space
                            l2_distance = torch.norm(
                                current_img - source_orig, p=2)
                            linf_distance = torch.norm(
                                current_img - source_orig, p=float("inf"))
                            print(
                                f'Finishing after {i + 1} steps - L2 distance: {l2_distance:.4f} - L-Inf distance: {linf_distance:.4f}')

                            optimized_file = f'{args.output_folder}/{input_file_name}_opt_{linf_distance:.4f}'
                            if args.output_folder != '':
                                save_images(source+delta, args.output_folder,
                                            f'{input_file_name}_opt_{linf_distance:.4f}')
                                save_images(delta, args.output_folder,
                                            f'{input_file_name}_delta')
                            if data == 'nsfw':
                                logger_data = [img, optimized_file + '.png', l2_distance.item(),
                                               linf_distance.item(), i + 1,
                                               target_hash_dict[target_hash_hex][1]]
                            else:
                                logger_data = [img, optimized_file + '.png', l2_distance.item(),
                                               linf_distance.item(), i + 1,
                                               target_path]

                            logger.add_line(logger_data)
                            break


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Perform neural collision attack.')
    parser.add_argument('--source', dest='source', type=str,
                        default='inputs/source.png', help='image to manipulate')
    parser.add_argument('--data', type=str,
                        default='coco', choices=['coco', 'mnist', 'nsfw'])
    parser.add_argument('--model_path', type=str,
                        default='/home/yuchen/code/verified_phash/train_verify/64_ep1_resv5_l1_aug2/ckpt_best.pth', help='path of model weight')
    parser.add_argument('--learning_rate', dest='learning_rate', default=1e-3,
                        type=float, help='step size of PGD optimization step')
    parser.add_argument('--optimizer', dest='optimizer', default='Adam',
                        type=str, help='kind of optimizer')
    parser.add_argument('--ssim_weight', dest='ssim_weight', default=10,
                        type=float, help='weight of ssim loss')
    parser.add_argument('--experiment_name', dest='experiment_name',
                        default='preimage_attack', type=str, help='name of the experiment and logging file')
    parser.add_argument('--output_folder', dest='output_folder',
                        default='collision_attack_outputs_robust', type=str, help='folder to save optimized images in')
    parser.add_argument('--target_hashset', dest='target_hashset',
                        type=str, help='Target hashset csv file path')
    parser.add_argument('--edges_only', dest='edges_only',
                        action='store_true', help='Change only pixels of edges')
    parser.add_argument('--sample_limit', dest='sample_limit',
                        default=10000, type=int, help='Maximum of images to be processed')
    parser.add_argument('--threads', dest='num_threads',
                        default=1000, type=int, help='Number of parallel threads')
    parser.add_argument('--check_interval', dest='check_interval',
                        default=10, type=int, help='Hash change interval checking')
    parser.add_argument('--epsilon',
                        default=16./255, type=float)

    args = parser.parse_args()

    model_path = args.model_path
    # Load and prepare components
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if args.data == 'nsfw':
        # Read target hashset
        target_hash_dict = dict()
        bin_hex_hash_dict = dict()
        target_hashes = []
        with open(args.target_hashset, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader, None)  # skip header
            for row in reader:
                hash_string = row[2].strip('][').split()
                hash = [int(b) for b in list(hash_string)]
                # hash = torch.tensor(hash).unsqueeze(0).to(device)
                target_hash_dict[row[3]] = [str(hash), row[1]]
                target_hash = torch.tensor(hash).unsqueeze(0)
                bin_hex_hash_dict[str(hash)] = row[3]
                target_hashes.append(target_hash)
            target_hashes = torch.cat(target_hashes, dim=0).to(device)


    if args.output_folder != '':
        try:
            os.mkdir(args.output_folder)
        except:
            if not os.listdir(args.output_folder):
                print(
                    f'Folder {args.output_folder} already exists and is empty.')
            else:
                print(
                    f'Folder {args.output_folder} already exists and is not empty, delete it')
                shutil.rmtree(args.output_folder)
                os.mkdir(args.output_folder)

    # Prepare logging
    logging_header = ['file', 'optimized_file', 'l2', 'l_inf','steps']
    logger = Logger(args.experiment_name, logging_header, output_dir=f'{args.output_folder}/logs')
    # logger.add_line(['Hyperparameter', args.source, args.learning_rate,
    #                  args.optimizer, args.ssim_weight, args.edges_only])

    # Load images
    if os.path.isfile(args.source):
        images = [args.source]
    elif os.path.isdir(args.source):
        images = [join(args.source, f) for f in os.listdir(
            args.source) if isfile(join(args.source, f))]
        images = sorted(images)
    else:
        raise RuntimeError(f'{args.source} is neither a file nor a directory.')
    # images = images[:args.sample_limit]
    if len(images) > args.sample_limit:
        images = random.sample(images, args.sample_limit)

    threads_args = (images, device,logger, args, model_path, args.epsilon, args.data)


    # for t in range(args.num_threads):
    #     optimization_thread(*threads_args)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1000) as executor:
        for t in range(args.num_threads):
            executor.submit(lambda p: optimization_thread(*p), threads_args)

    logger.finish_logging()

    '''
    Statistic
    '''
    data = []
    columns = ['file', 'optimized_file', 'l2', 'l_inf', 'steps', 'source']
    with open(f'{args.output_folder}/logs/{args.experiment_name}.csv', 'r') as file:
        for line in file:
            line = line.strip()
            parts = line.split(',')
            if len(parts) == len(columns):
                data.append(parts)
            else:
                print("Skipped line:", line)

    df = pd.DataFrame(data, columns=columns)
    print(f'l2: {np.mean([float(i) for i in df.loc[:, "l2"].tolist()])}')
    print(f'l_inf: {np.mean([float(i) for i in df.loc[:, "l_inf"].tolist()])}')
    print(f'steps: {np.mean([float(i) for i in df.loc[:, "steps"].tolist()])}')
    df['l_inf'] = df['l_inf'].astype(float)
    epsilon = "{:.4f}".format(max(df['l_inf']))
    print(f"===>Collision Rate under {epsilon}: {(len(df) / args.sample_limit * 100)}%")


if __name__ == "__main__":
    os.makedirs('./temp', exist_ok=True)
    main()
