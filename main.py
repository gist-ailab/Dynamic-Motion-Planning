
from os.path import join

from planners import MPNetPlanner
from environments import Dynamic2D, MPNetSimple2D

from file_utils import get_file_list
import random
import numpy as np
import time

if __name__ == '__main__':
    mpnet_data_root = "./dataset"
    S2D_data_path = join(mpnet_data_root, "obs_cloud")
    obc_file_list = []
    for i in range(10):
        target_path = join(S2D_data_path, "obc{}.dat".format(0 + i))
        obc_file_list.append(target_path)
    cae_weight = "cae_weight.tar"
    # mlp_weight = "epoch_195.tar"
    mlp_weight = "results/2021-02-23_12-57-36/epoch_200.tar"
    planner = MPNetPlanner(cae_weight, mlp_weight)
    print(planner.encoder, planner.planner)

    for i in range(20):
        target_path = random.choice(obc_file_list)
        env = MPNetSimple2D(obc_file=target_path)
        print(env)
        planner.reset(env)
        env.decoder_view(planner.obs_dec)
        print(planner.obs_dec)
        for j in range(100):
            next_config = planner.get_next_config()
            env.visualize_with_decodedview(next_config)
            if planner.is_reaching_target():
            #     time.sleep(2)
                break

