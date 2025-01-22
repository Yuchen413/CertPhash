import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
import pandas as pd
import multiprocessing as mp
from tqdm.notebook import tqdm
import argparse
from benign0_func_check import get_rotation_angles, get_translation_tuples, get_rotation_angles_positive

def l1_distance(tensor1, tensor2):
    loss = torch.nn.L1Loss(reduction='sum')
    l1 = []
    for i in range(tensor1.shape[0]):
        l1.append(loss(tensor1[i], tensor2[i]).item())
    return l1


def l1_distance_pairwise(tensor1, tensor2, batch_size=200, use_gpu=True):
    if use_gpu:
        tensor1, tensor2 = tensor1.cuda(), tensor2.cuda()

    num_items_1 = tensor1.size(0)
    num_items_2 = tensor2.size(0)
    l1_matrix = torch.zeros((num_items_1, num_items_2), device='cuda' if use_gpu else 'cpu')

    # Process in batches
    for i in range(0, num_items_1, batch_size):
        end_i = i + batch_size
        tensor1_batch = tensor1[i:end_i]

        for j in range(0, num_items_2, batch_size):
            end_j = j + batch_size
            tensor2_batch = tensor2[j:end_j]

            # Expand and compute distances
            tensor1_expanded = tensor1_batch.unsqueeze(1)
            tensor2_expanded = tensor2_batch.unsqueeze(0)
            diff = torch.abs(tensor1_expanded - tensor2_expanded)
            l1_distances_batch = diff.sum(2)

            # Assign computed distances to the appropriate submatrix
            l1_matrix[i:end_i, j:end_j] = l1_distances_batch

    return l1_matrix.cpu()


def calculate_roc_auc(l1_matrix, RANGE, num_thresholds=20, steepness=0.01, batch_size=100):
    l1_matrix = torch.tensor(l1_matrix, device='cuda')  # Move the matrix to GPU
    n = l1_matrix.size(0)
    threshold_range = RANGE

    # Initialize storage for all probabilities and labels
    all_probabilities = []
    all_labels = []

    # Process the full matrix in batches
    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)
        batch = l1_matrix[i:end_i, i:end_i]
        should_match = torch.eye(batch.size(0), device='cuda', dtype=torch.bool)

        # Process each threshold
        thresholds = np.linspace(threshold_range[0], threshold_range[1], num=num_thresholds)
        for threshold in thresholds:
            # Convert distances to probabilities
            probabilities = 1 / (1 + torch.exp(-steepness * (threshold - batch)))

            # Store probabilities and labels
            all_probabilities.append(probabilities.flatten())
            all_labels.append(should_match.flatten())

    # Concatenate results
    all_probabilities = torch.cat(all_probabilities).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(all_labels, all_probabilities)
    roc_auc = auc(fpr, tpr)

    #calculate fpr an tpr with 0.1
    predictions = (all_probabilities >= 0.1).astype(int)
    TP = np.sum((predictions == 1) & (all_labels == 1))
    FP = np.sum((predictions == 1) & (all_labels == 0))
    TN = np.sum((predictions == 0) & (all_labels == 0))
    FN = np.sum((predictions == 0) & (all_labels == 1))
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0


    return FPR, TPR, thresholds, roc_auc


def get_hashes_from_csv(file_path):
    df = pd.read_csv(file_path)
    if 'pdq' in TARGET:
        df['hash_bin'] = df['hash_bin'].apply(lambda x: list(map(int, x.strip('[]').split())))
        bin_hashes = np.array(df['hash_bin'].tolist()).astype(int)
    elif 'neuralhash' in TARGET:
        bin_hashes = []
        for bit_string in df['hash_bin']:
            bin_hashes.append(list(bit_string))
        bin_hashes = np.array(bin_hashes, dtype=int)
    else:
        df['hash_bin'] = df['hash_bin'].apply(lambda x: x.strip('[]').split() if 'nn' in TARGET else x.split())
        df['hash_bin'] = df['hash_bin'].apply(lambda x: [int(i) for i in x])
        bin_hashes = np.array(df['hash_bin'].tolist()).astype(int)
    # print(bin_hashes)
    return bin_hashes


def get_augmented_hashes_and_l1_dist(filenames, augment_values, original_hashes, return_bin_hashes=False,
                                     num_processes=2, pairwise=True, ):
    return_hash_dict = {}
    return_l1_dict = {}

    # with tqdm(total=len(filenames)) as pbar:
    with mp.Pool(num_processes) as pool:
        for i, bin_hashes in tqdm(enumerate(pool.imap(get_hashes_from_csv, filenames))):
            return_hash_dict[augment_values[i]] = bin_hashes
            if pairwise == False:
                # Only calculate the l1 between same index
                return_l1_dict[augment_values[i]] = np.array(l1_distance(torch.tensor(bin_hashes), original_hashes))
            else:
                return_l1_dict[augment_values[i]] = np.array(
                    l1_distance_pairwise(torch.tensor(bin_hashes), original_hashes))

    if return_bin_hashes:
        return return_hash_dict, return_l1_dict

    return return_l1_dict


def print_mean_and_std_for_keys(given_dict):
    for key in given_dict.keys():
        print(f'Mean L1 Distance for {key}: {given_dict[key].mean()}')
        print(f'Standard Deviation L1 Distance for {key}: {given_dict[key].std()}')

def process_transformation(trans_name, values, bin_hashes_orig, args, num_processes):
    """
    General function to process a transformation and calculate ROC AUC.
    """
    if 'flip' in trans_name:
        file_paths = [
            os.path.join(args.hash_dir, f'{trans_name}', f'{args.dataset}_{trans_name}.csv') for value in values
        ]
    else:
        file_paths = [
            os.path.join(args.hash_dir, f'{trans_name}', f'{args.dataset}_{trans_name}_{value}.csv') for value in values
        ]
    distances = get_augmented_hashes_and_l1_dist(
        file_paths, values, bin_hashes_orig, num_processes=num_processes, pairwise=True
    )

    keys = []
    auc_values = []
    fpr_values = []
    tpr_values = []
    for idx, (key, value) in enumerate(tqdm(distances.items())):
        fpr, tpr, thresholds, roc_auc = calculate_roc_auc(value, args.range)
        keys.append(key)
        auc_values.append(roc_auc)
        fpr_values.append(fpr)
        tpr_values.append(tpr)

    save_results(keys, auc_values, fpr_values, tpr_values, args, trans_name)
    return keys, auc_values, fpr_values, tpr_values

def save_results(keys, auc_values,fpr_values, tpr_values, args, trans_name):
    """
    Save results to a file, print them, and also calculate mean and standard deviation of AUC values.
    """
    mean_auc = np.mean(auc_values)
    std_auc = np.std(auc_values)
    mean_fpr = np.mean(fpr_values)
    std_fpr = np.std(fpr_values)
    mean_tpr = np.mean(tpr_values)
    std_tpr = np.std(tpr_values)

    # Display transformation name and ROC AUC values
    print(f"{trans_name}: ", end='')
    for key in keys:
        print(f"{key:<10}", end='')
    print()

    print("ROC AUC: ", end='')
    for value in auc_values:
        print(f"{value:<10.4f}", end='')
    print()

    # print("FPR: ", end='')
    # for value in fpr_values:
    #     print(f"{value:<10.4f}", end='')
    # print()
    #
    # print("TPR: ", end='')
    # for value in tpr_values:
    #     print(f"{value:<10.4f}", end='')
    # print()

    print(f"Mean ROC AUC: {mean_auc:.4f}, Std ROC AUC: {std_auc:.4f}\n")
    # print(f"Mean FPR: {mean_fpr:.4f}, Std FPR: {std_fpr:.4f}\n")
    # print(f"Mean TPR: {mean_tpr:.4f}, Std TPR: {std_tpr:.4f}\n")

    # Append data to a file
    with open(f'{args.plot_dir}/{args.dataset}_results.txt', 'a') as file:
        file.write(f"{trans_name}:  ")
        for key in keys:
            file.write(f"{key:<10}")
        file.write("\n")

        file.write("ROC AUC: ")
        for value in auc_values:
            file.write(f"{value:<10.4f}")
        file.write("\n")

        file.write(f"Mean ROC AUC: {mean_auc:.4f}, Std ROC AUC: {std_auc:.4f}\n\n")
        # file.write(f"Mean FPR: {mean_fpr:.4f}, Std FPR: {std_fpr:.4f}\n\n")
        # file.write(f"Mean TPR: {mean_tpr:.4f}, Std TPR: {std_tpr:.4f}\n\n")


def setup_parser():
    parser = argparse.ArgumentParser(description="Process some datasets and transformations.")

    # Adding arguments to the parser
    parser.add_argument('--dataset', type=str, default='coco_val', help='args.dataset to process')
    parser.add_argument('--target', type=str, default='photodna_nn_cert', help='shold be the same as it for benign0_func_check.py')
    parser.add_argument('--max_rot_angle', type=int, default=64, help='Maximum rotation angle')
    parser.add_argument('--rot_log_base', type=int, default=2, help='Base for logarithmic rotation calculation')
    parser.add_argument('--rot_steps', type=int, default=7, help='Number of rotation steps')

    return parser

def process_arguments(args):
    if 'photodna' in args.target:
        args.range = [1600, 2000]
    elif 'pdq' in args.target:
        args.range = [60, 120]
    else:
        args.range = [10, 30]

    args.plot_dir = f'func_logs/{args.dataset}_{args.target}'
    args.hash_dir = f'func_logs/{args.dataset}_{args.target}'

    # Create directories if they do not exist
    os.makedirs(args.plot_dir, exist_ok=True)

    return args

def main():
    global TARGET
    global RANGE
    parser = setup_parser()
    args = parser.parse_args()
    args = process_arguments(args)
    TARGET = args.target
    RANGE = args.range


    bin_hashes_orig = torch.tensor(get_hashes_from_csv(os.path.join(args.hash_dir, f'{args.dataset}_original.csv')))

    # Process various transformations
    transformations = {
        'rotation': (get_rotation_angles_positive(args.max_rot_angle, args.rot_log_base, args.rot_steps), 15),
        'hue': (list(range(-180, 180, 30)), 12),
        'brightness': (list(np.linspace(0, 2, 9, endpoint=True)), 9),
        'contrast': (list(np.linspace(0, 2, 9, endpoint=True)), 9),
        'saturation': (list(np.linspace(0, 2, 9, endpoint=True)), 9),
        'compression': ([100] + list((100 - np.ceil(np.logspace(0, np.log(100) / np.log(1.5), 10, endpoint=True, base=1.5))).clip(0, 100)), 11),
        'crop': (list(filter(lambda x: x != 63, [64] + list(64 - np.append(np.logspace(0, 5, 6, base=2, endpoint=True, dtype=int), [32])))), 10),
        'hflip': ([0], 1),
        'vflip': ([0], 1),
        'downsizing': (list(filter(lambda x: x != 63, [64] + list(64 - np.append(np.logspace(0, 5, 6, base=2, endpoint=True, dtype=int), [32])))),9),
        'perceptive': (list([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]), 9),
        'gaussianBlur': (list([0, 1, 2, 3, 4, 5, 6, 8, 10]), 9),
        'erasing': (list(np.linspace(0, 0.3, 6, endpoint=True)), 9),
        'gray': (list(np.linspace(0, 1, 9, endpoint=True)),9),
        'resizing': (list([100, 80, 64, 56, 48, 32]),6),
        'pepperSalt': (list([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]), 9)
    }

    for trans_name, (values, num_processes) in transformations.items():
        process_transformation(trans_name, values, bin_hashes_orig, args, num_processes)


if __name__ == "__main__":
    main()