
from os.path import join

# from planners import MPNetPlanner
# from environments import Dynamic2D, MPNetSimple2D
from pyrep_environments import Plane2D

from file_utils import get_file_list
import random
import numpy as np
import time

def main():
    mpnet_data_root = "./dataset"
    mpnet_data_root = "/home/raeyo/dev_tools/MotionPlanning/MPNet/MPNetDataset/S2D/dataset"
    cloud_data_path = join(mpnet_data_root, "obs_cloud")
    obc_file_list = []
    for i in range(10):
        target_path = join(cloud_data_path, "obc{}.dat".format(100 + i))
        obc_file_list.append(target_path)
    path_file_list = []
    for i in range(10):
        target_path = join(mpnet_data_root, "e{}".format(100 + i))
        path_list = get_file_list(target_path)
        path_file_list.append(path_list)
    
    cae_weight = "pretrained_weights/cae_weight.tar"
    mlp_weight = "pretrained_weights/mlp_weight_495.tar"
    planner = MPNetPlanner(cae_weight, mlp_weight)
    
    for i in range(100):
        target_idx = random.randint(0,9)
        target_obc = obc_file_list[target_idx]
        target_path = random.choice(path_file_list[target_idx])
        env = MPNetSimple2D(obc_file=target_obc, path_file=target_path)
        planner.reset(env)
        env.set_decoder_view(planner.obs_dec)
        # for j in range(100):
        #     next_config = planner.get_next_config()
        #     env.visualize_with_decodedview(next_config)
        #     planner.update_current_config(next_config)
        #     if planner.is_reaching_target():
        #         break
        # if planner.is_reaching_target():
        #     print("Success to reach target")
        # else:
        #     print("Fail to reach target")
        

        path = planner.planning()
    
        if path is not None:
            for conf in path:
                env.visualize_with_decodedview(conf)
            print("Success to reach target")
        else:
            print("Fail to reach target")


if __name__ == '__main__':
    env = Plane2D()
    episode_num = 10
    episode_length = 1000
    for i in range(episode_num):
        obstacle_num = np.random.randint(10)
        env.reset(obstacle_num=obstacle_num)

        for j in range(episode_length):
            env.step()

            

