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
    def __init__(self, args):
        # load pretrained encoder
        self.encoder = Encoder().eval()
        assert args.cae_weight, "No pretrained cae weight"
        self.encoder.load_state_dict(torch.load(args.cae_weight), strict=False)

        # load pretrained planner
        self.planner = MLP(args.input_size, args.output_size).eval()
        assert args.load_weights, "No pretrained mlp weight"
        self.planner.load_state_dict(torch.load(args.mlp_weights))

        # get obstacle cloud and representation
        self.obs_rep = self.get_obs_representation() # 28
        
        # planning config
        self.start = None
        self.goal = None
        self.current = self.start

    def get_obs_representation(self):
        obs_input = torch.from_numpy(self.obc).float()
        obs_rep = self.encoder(obs_input)
        return obs_rep.cpu().detach().numpy()

    def get_start_and_goal_config(self):
        start, goal = np.zeros(2), np.zeros(2)

        return start, goal
    
    def get_collision_state(self, config):
        is_collision = False
        for i in range(OBS_NUM):
            for j in range(NUM_DIM):
                dist = abs(self.obc[i][j] - config[j])
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
                if self.get_collision_state(current_state):
                    return False
                
                current_state = current_state + dif

            if self.get_collision_state(end):
                return False

        return True

    def is_reaching_target(self):
        current, goal = None, None
        distance = self.get_distance(current, goal)
        if distance > 1.0:
            return False
        else:
            return True

    def lazy_vertex_contraction(self, path, idx):
        for i in range(0,len(path)-1):
            for j in range(len(path)-1,i+1,-1):
                ind=0
                ind=self.steerTo(path[i],path[j],idx)
                if ind==1:
                    pc=[]
                    for k in range(0,i+1):
                        pc.append(path[k])
                    for k in range(j,len(path)):
                        pc.append(path[k])

                    return self.lazy_vertex_contraction(pc,idx)
                    
	    return path

    def feasibility_check(self, path, idx):
        for i in range(len(path) -1):
            tag = self.steerTo(path[i], path[i + 1], idx)
            if not tag:
                return False
        return True

    def collision_check(self, path, idx):
        for p in path:
            if self.get_collision_state(p, idx):
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
    

    def bidirectional_planning(self, start, goal):

        pass

    def plan(self, env, start, goal):
        env.get_obstacle_cloud()
        

