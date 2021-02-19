from torch._C import dtype
from torch.utils.data import Dataset
import torch
from torch.autograd import Variable

import numpy as np
import os

from fileApi import get_file_list

from os.path import join

class ObstacleDataSet(Dataset):
    def __init__(self, root, is_val=False, is_test=False):
        super(ObstacleDataSet, self).__init__()
        if is_val:
            self.data_root = root + '/val'
        elif is_test:
            self.data_root = root + '/test'
        else:
            self.data_root = root + '/train'
        self.file_names = get_file_list(self.data_root)
        self.file_names.sort()

    def __len__(self):
        return len(self.file_names)

    @staticmethod
    def transform(nd_array):

        return torch.from_numpy(nd_array).float()

    def __getitem__(self, idx):
        obstacle_file = self.file_names[idx]
        obs_np = np.fromfile(obstacle_file)

        return self.transform(obs_np)

class PathDataSet(Dataset):
    def __init__(self, root, encoder, is_val=False):
        super(PathDataSet, self).__init__()
        
        self.data_root = join(root, "dataset")
        if is_val:
            self.st_idx = 100
            self.end_idx = self.st_idx + 10
        else:
            self.st_idx = 0
            self.end_idx = self.st_idx + 100
        self.encoder = encoder # pretrained
        self.dim = 2

        self.obs2path = {}
        self.obs_representation = self.get_obs_representation()
        self.load_path_list()

        self.dataset = self.create_dataset()
        
        self.file_names = get_file_list(self.data_root)
        self.file_names.sort()

    def get_obs_representation(self):
        obs_root = join(self.data_root, "obs_cloud")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        obs_representations = {}
        for obs_idx in range(self.st_idx, self.end_idx):
            self.obs2path.setdefault(obs_idx, [])
            target_file = join(obs_root, "obc{}.dat".format(obs_idx))
            obs_np = np.fromfile(target_file)
            obs_input = self.transform(obs_np)
            obs_input = obs_input.to(device)
            obs_rep = self.encoder(obs_input)
            obs_rep = obs_rep.cpu().detach().numpy()
            obs_representations[obs_idx] = obs_rep
        
        return obs_representations

    def load_path_list(self):
        for obs_idx in range(self.st_idx, self.end_idx):
            path_root = join(self.data_root, "e{}".format(obs_idx))
            path_file_list = get_file_list(path_root)
            for path_file in path_file_list:
                path = np.fromfile(path_file)
                path = path.reshape(-1, self.dim) # n, (x, y)
                self.obs2path[obs_idx].append(path)
    
    def create_dataset(self):
        dataset = []
        for obs_idx in self.obs2path.keys():
            obs_rep = self.obs_representation[obs_idx] # 28
            path_list = self.obs2path[obs_idx] # Num_path, Len_path, 2
            for path in path_list:
                if not len(path) > 0:
                    continue
                goal_conf = path[-1]
                for cur_idx, point in enumerate(path[:-1]):
                    cur_conf = point
                    next_conf = path[cur_idx + 1]
                    dataset.append({
                        "obstacle_representation": obs_rep,
                        "goal_config": goal_conf,
                        "current_config": cur_conf,
                        "next_config": next_conf
                    })

        return dataset     

    @staticmethod
    def transform(nd_array):
        return torch.from_numpy(nd_array).float()


    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        input_data = []
        target_data = self.dataset[idx]
        input_data.append(target_data["obstacle_representation"]) # 28
        input_data.append(target_data["current_config"]) # 2
        input_data.append(target_data["goal_config"]) # 2
        
        input_data = np.concatenate(input_data)
        input_data = self.transform(input_data)
        next_conf = self.transform(target_data["next_config"]) # 2

        return input_data, next_conf
        


