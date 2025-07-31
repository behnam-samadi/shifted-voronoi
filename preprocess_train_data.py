import os
import numpy as np
import torch
from tqdm import tqdm

# Make sure you import or define your proposed_grouping function
from partial_tests.kdtree_based_partitioning import proposed_grouping, estimate_max_k  # example import path, update if needed

def preprocess_and_save(data_root, save_root, split='train', test_area=5):
    os.makedirs(save_root, exist_ok=True)

    # Use your dataset filtering logic for file list
    data_list = sorted(os.listdir(data_root))
    data_list = [item for item in data_list if 'Area_' in item and item.endswith('.npy')]

    if split == 'train':
        data_list = [item for item in data_list if f'Area_{test_area}' not in item]
    else:
        data_list = [item for item in data_list if f'Area_{test_area}' in item]

    print(f"Preprocessing {len(data_list)} files for split '{split}' excluding Area_{test_area}")

    for item in tqdm(data_list, desc='Preprocessing'):
        data_path = os.path.join(data_root, item)
        data = np.load(data_path)

        coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]

        points = torch.from_numpy(coord).to('cuda')
        # Run your expensive preprocessing here
        downsample_rate = 0.1
        optimum_threshold = estimate_max_k(coord.shape[0], int(coord.shape[0] * downsample_rate))
        groups = proposed_grouping(points, optimum_threshold)

        save_path = os.path.join(save_root, item.replace('.npy', '.pt'))
        torch.save({
            'coord': torch.from_numpy(coord).float(),
            'feat': torch.from_numpy(feat).float(),
            'label': torch.from_numpy(label).long(),
            'groups': groups,
        }, save_path)

    print(f"Preprocessing complete. Saved to {save_root}")

if __name__ == '__main__':
    data_root = '/home/samadi/research/pythonProject/data/stanford_indoor3d/'  # your raw data folder
    save_root = '/home/samadi/research/pythonProject/data/s3dis/preprocessed/'  # folder to save preprocessed files
    preprocess_and_save(data_root, save_root, split='train', test_area=5)
