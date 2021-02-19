
from os.path import join

from planners import MPNetPlanner
from environments import Dynamic2D, MPNetSimple2D

from file_utils import get_file_list
import random
import numpy as np


if __name__ == '__main__':
    mpnet_data_root = "/home/raeyo/dev_tools/MotionPlanning/MPNet/MPNetDataset"
    S2D_data_path = join(mpnet_data_root, "S2D/dataset/obs_cloud")
    obc_file_list = []
    for i in range(10):
        target_path = join(S2D_data_path, "obc{}.dat".format(100 + i))
        obc_file_list.append(target_path)
    cae_weight = "cae_weight.tar"
    mlp_weight = "mlp_weight.tar"
    planner = MPNetPlanner(cae_weight, mlp_weight)

    for i in range(100):
        target_path = random.choice(obc_file_list)
        env = MPNetSimple2D(obc_file=target_path)
        planner.reset(env)
        for j in range(100):
            next_config = planner.get_next_config()
            env.visualize(next_config)

