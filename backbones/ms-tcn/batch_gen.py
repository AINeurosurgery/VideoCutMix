#!/usr/bin/python2.7

import torch
import numpy as np
import random
from typing import Dict
import yaml

def read_dataset_config(config_path: str = "./configs/config.yaml") -> Dict[str, any]:
    """
    Args:
        config_path: Path for dataset config to load names of datasets
    """

    with open(config_path, "r") as f:
        data = yaml.load(f, Loader = yaml.FullLoader)
    return data["datasets"]


class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate, epoch_level_augmentation=False, epoch_variational_gt = False):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.epoch_level_augmentation = epoch_level_augmentation
        self.epoch_variational_gt = epoch_variational_gt

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        random.shuffle(self.list_of_examples)

    def next_batch(self, batch_size, flag=None, epoch=-1):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        for vid in batch:
            vid_name = vid.replace(".txt", "")
            if not self.epoch_level_augmentation:
                features = np.load(self.features_path + vid_name + '.npy')  # dim: 2048 x frame#
            else:
                features = np.load(self.features_path + str(epoch + 1) + f'/{vid_name}.npy')
                
            if not self.epoch_variational_gt:
                file_ptr = open(self.gt_path + vid, 'r')
            else:
                file_ptr = open(self.gt_path + str(epoch + 1) + f"/{vid}", 'r')
            
            content = file_ptr.read().split('\n')[:-1]
            if self.epoch_variational_gt and flag != 'target':
                classes = np.zeros((min(np.shape(features)[1], len(content)), self.num_classes))
                for i in range(len(classes)):
                    data = list(map(lambda x: float(x), content[i].split(" ")))
                    for j in range(len(classes[i])):
                        classes[i][j] = data[j]
                batch_input.append(features[:, ::self.sample_rate])
                batch_target.append(classes[::self.sample_rate, :])
            else:
                classes = np.zeros(min(np.shape(features)[1], len(content)))  # ground truth (in indices)
                for i in range(len(classes)):
                    classes[i] = self.actions_dict[content[i]]
                batch_input.append(features[:, ::self.sample_rate])
                batch_target.append(classes[::self.sample_rate])

        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        if self.epoch_variational_gt and flag != "target":
            batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), self.num_classes, dtype=torch.float)*(-100)
        else:
            batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i][:, :max(length_of_sequences)])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, mask
