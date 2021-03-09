import torch
from torch.autograd import Variable 

import numpy as np
import math
import copy

import torch.nn as nn
from .MPNet_models import MLP, Encoder, Decoder, CAE

import time

import matplotlib.pyplot as plt

OBS_NUM = 7
NUM_DIM = 2 # for x, y
OBS_SIZE = 5
DISCRETIZATION_STEP=0.01

class MPNetPlanner():
    def __init__(self, cae_weight, mlp_weight):
        # load pretrained encoder
        self.encoder = Encoder().eval()
        self.encoder.load_state_dict(torch.load(cae_weight), strict=False)

        self.decoder = Decoder().eval()
        self.decoder.load_state_dict(torch.load(cae_weight), strict=False)

        self.CAE = CAE().eval()
        self.CAE.load_state_dict(torch.load(cae_weight))

        # load pretrained planner
        self.planner = MLP(32, 2)
        self.planner.load_state_dict(torch.load(mlp_weight))

        self.env = None
        self.start, self.end, self.current = None, None, None

    def get_obs_representation(self):
        obs_input = torch.from_numpy(self.obs_cloud).float()
        obs_rep = self.CAE.encode(obs_input)
        return obs_rep.cpu().detach().numpy()

    def get_obs_decoded_view(self):
        obs_input = torch.from_numpy(self.obs_cloud).float()
        obs_output = self.CAE(obs_input)
        return obs_output.cpu().detach().numpy().reshape(-1, 2)

    def check_collision(self, config):
        for i in range(OBS_NUM):
            is_collision = True
            for j in range(NUM_DIM):
                dist = abs(self.obstacles_xy[i][j] - config[j])
                if dist > OBS_SIZE / 2.0:
                    is_collision = False
                    break
            if is_collision:
                return is_collision
        return is_collision

    def steerTo(self, start, end):
        start = np.array(start)
        end = np.array(end)
        dif = end - start
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
            if self.steerTo(self.current, self.end):
                return True
            return False
        else:
            return True

    def lazy_vertex_contraction(self, path):
        contracted_path = []
        for start_idx in range(0, len(path)-1): # 0, 1, 2, 3
            for end_idx in range(len(path)-1, start_idx+1,-1): # 3, 2
                target_reached=False
                target_reached=self.steerTo(path[start_idx], path[end_idx])
                if target_reached:
                    for k in range(0, start_idx+1):
                        contracted_path.append(path[k])
                    for k in range(j,len(path)):
                        contracted_path.append(path[k])

                    return self.lazy_vertex_contraction(contracted_path)
                    
        return path

    def feasibility_check(self, path):
        for i in range(len(path) - 1):
            tag = self.steerTo(path[i], path[i + 1])
            if tag==False:
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
        distance = np.linalg.norm(dif)
        return distance

    def reset(self, env):
        self.env = env

        # get obstacle cloud and representation
        self.obs_cloud = self.env.get_obstacle_cloud()
        self.obs_rep = self.get_obs_representation() # 28
        self.obstacles_xy = self.env.obstacles_xy
        self.obs_dec = self.get_obs_decoded_view()

        # planning config
        self.start, self.end = self.env.get_start_and_end()
        self.current = self.start

    def planning(self):
        assert self.current == self.start, "current config is not start state"
        
        path, target_reached = self.get_path(self.start, self.end, max_itr=80)
        
        if target_reached:
            pass
        
        path = self.lazy_vertex_contraction(path)
        is_feasible = self.feasibility_check(path)

        if not is_feasible:
            replan_step = 0
            is_feasible = False
            while not is_feasible and replan_step < 10:
                replan_step += 1
                path = self.replan(path)
                if path is not None:
                    path = self.lazy_vertex_contraction(path)
                    is_feasible = self.feasibility_check(path)
                else:
                    break

                if is_feasible:
                    return path

        else:
            return path
        

    def get_path(self, start, goal, max_itr=50):
        path = []
        path.append(start)
        current = start
        target_reached = False
        step = 0
        while not target_reached and step < max_itr:
            step += 1
            next_conf = self._get_next_config(current, goal)
            path.append(next_conf)        
            current = next_conf    
            target_reached = self.steerTo(current, goal)

        path.append(copy.deepcopy(goal))
        
            
        return path, target_reached
           
    def replan(self, path):
        # check collision config in previous path
        noncollision_path = []
        noncollision_path.append(path[0])
        for config in path[1:-1]:
            if not self.check_collision(config):
                noncollision_path.append(config)

        noncollision_path.append(path[-1])    

        new_path = []
        for i in range(0, len(noncollision_path)-1):
            start = noncollision_path[i]
            goal = noncollision_path[i+1]
            steer = self.steerTo(start, goal)

            if steer:
                new_path.append(start)
                new_path.append(goal)
            else:
                path, target_reached = self.get_path(start, goal, max_itr=50)
                if target_reached:
                    new_path += path
                else:
                    new_path += path

        return new_path
                
    def get_next_config(self):
        is_collision = True
        step = 0
        next_conf = None
        while is_collision and step < 10:
            step += 1
            next_conf = self._get_next_config(self.current, self.end)
            is_collision = self.check_collision(next_conf)
        
        return next_conf

    def _get_next_config(self, current, goal):
        assert self.env, "No workspace"
        obs_rep = torch.Tensor(self.obs_rep)
        cur_conf = torch.Tensor(current)
        end_conf = torch.Tensor(goal)        
        
        planner_inp = torch.cat((obs_rep, cur_conf, end_conf))
        next_conf = self.planner(planner_inp)
        
        return next_conf.detach().numpy()

    def update_current_config(self, config):
        self.current = config


