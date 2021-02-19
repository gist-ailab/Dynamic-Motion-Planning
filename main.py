
from environments import Dynamic2D, MPNetSimple2D
from os.path import join
# from planners import Planner

from file_utils import get_file_list







if __name__ == '__main__':
    mpnet_data_root = "/home/raeyo/dev_tools/MotionPlanning/MPNet/MPNetDataset"
    S2D_data_path = join(mpnet_data_root, "S2D/dataset/obs_cloud")
    obc_file_list = get_file_list(S2D_data_path)

    env = MPNetSimple2D()

    for i in range(10000):
        env.visualize(i)

