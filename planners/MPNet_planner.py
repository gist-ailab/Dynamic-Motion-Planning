import torch
from torch.autograd import Variable 

import numpy as np
import math
import copy

from .MPNet.MPNet_models import MLP, Encoder

OBS_NUM = 7
NUM_DIM = 2 # for x, y
OBS_SIZE = 5
DISCRETIZATION_STEP=0.01

class MPNetPlanner():
    def __init__(self, cae_weight, mlp_weight):
        # load pretrained encoder
        self.encoder = Encoder().eval()
        self.encoder.load_state_dict(torch.load(cae_weight), strict=False)

        # load pretrained planner
        self.planner = MLP(32, 2).eval()
        self.planner.load_state_dict(torch.load(mlp_weight))

        self.env = None
        self.start, self.end, self.current = None, None, None

    def get_obs_representation(self):
        obs_input = torch.from_numpy(self.obs_cloud).float()
        obs_rep = self.encoder(obs_input)
        return obs_rep.cpu().detach().numpy()

    def check_collision(self, config):
        is_collision = False
        for i in range(OBS_NUM):
            for j in range(NUM_DIM):
                dist = abs(self.obstacles_xy[i][j] - config[j])
                if dist > OBS_SIZE / 2.0:
                    break
            if is_collision:
                return is_collision
        return is_collision

    def steerTo(self, start, end):
        dif = np.array(end) - np.array(start)
        dist = np.linalg.norm(dif)

        if dist > 0:
            incrementTotal = dist / DISCRETIZATION_STEP
            dif = dif / incrementTotal
            
            numSegments = int(math.floor(incrementTotal))

            current_state = copy.deepcopy(start)
            for i in range(numSegments):
                if self.check_collision(current_state):
                    return False
                
                current_state = current_state + dif

            if self.check_collision(end):
                return False

        return True

    def is_reaching_target(self):
        distance = self.get_distance(self.current, self.end)
        if distance > 1.0:
            return False
        else:
            return True

    def lazy_vertex_contraction(self, path):
        for i in range(0,len(path)-1):
            for j in range(len(path)-1,i+1,-1):
                ind=False
                ind=self.steerTo(path[i],path[j])
                if ind:
                    pc=[]
                    for k in range(0,i+1):
                        pc.append(path[k])
                    for k in range(j,len(path)):
                        pc.append(path[k])

                    return self.lazy_vertex_contraction(pc)
                    
        return path

    def feasibility_check(self, path):
        for i in range(len(path) -1):
            tag = self.steerTo(path[i], path[i + 1])
            if not tag:
                return False
        return True

    def collision_check(self, path):
        for p in path:
            if self.check_collision(p):
                return False
        return True

    @staticmethod
    def to_var(x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    @staticmethod
    def get_distance(x, y):
        dif = np.array(x) - np.array(y)
        distance = np.norm(dif)
        return distance

    def reset(self, env):
        self.env = env

        # get obstacle cloud and representation
        self.obs_cloud = self.env.get_obstacle_cloud()
        self.obs_rep = self.get_obs_representation() # 28
        self.obstacles_xy = self.env.obstacles_xy

        # planning config
        self.start, self.end = self.env.get_start_and_end()
        self.current = self.start

    def get_next_config(self):
        assert self.env, "No workspace" 
        obs_rep = torch.Tensor(self.obs_rep)
        cur_conf = torch.Tensor(self.current)
        end_conf = torch.Tensor(self.end)
        
        planner_inp = torch.cat((obs_rep, cur_conf, end_conf))
        next_conf = self.planner(planner_inp)
        self.current = next_conf.detach().numpy()
        return self.current

