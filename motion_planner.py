import argparse
import copy
import torch
from torch._C import dtype
import torch.nn as nn
import numpy as np
import os
import pickle
from MPNet_models import MLP 
from torch.autograd import Variable 
import math
import time

OBS_NUM = 7
NUM_DIM = 2 # for x, y
OBS_SIZE = 5
DISCRETIZATION_STEP=0.01

class MPNetPlanner():
    def __init__(self, args):
        self.planner = MLP(32, 2)

        self.obc = None

        if args.load_weights:
            self.planner.load_state_dict(torch.load(args.load_weights))
        if torch.cuda.is_available():
            self.planner.cuda()
        
    def get_collision_state(self, x, idx):
        is_collision = False
        for i in range(OBS_NUM):
            for j in range(NUM_DIM):
                dist = abs(self.obc[idx][i][j] - x[j])
                if dist > OBS_SIZE / 2.0:
                    break
            if is_collision:
                return is_collision
        return is_collision

    def steerTo(self, current, end, idx):
        dif = np.array(end) - np.array(current)
        dist = np.linalg.norm(dif)

        if dist > 0:
            incrementTotal = dist / DISCRETIZATION_STEP
            dif = dif / incrementTotal
            
            numSegments = int(math.floor(incrementTotal))

            current_state = copy.deepcopy(current)
            for i in range(numSegments):
                if self.get_collision_state(current_state, idx):
                    return False
                
                current_state = current_state + dif

            if self.get_collision_state(end, idx):
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
    


