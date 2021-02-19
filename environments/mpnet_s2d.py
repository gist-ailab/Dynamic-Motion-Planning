import matplotlib.pyplot as plt
import numpy as np
from .entities import Zone, Obstacle
from .environment import Environment

class MPNetSimple2D(Environment):
    def __init__(self, obc_file):
        self.obc = np.fromfile(obc_file) # 2800
        self.obc_2d = np.reshape(self.obc, (-1, 2)) # 1400, 2

        # set workspace
        self.obc_min = np.min(self.obc_2d, axis=0) # 2
        self.obc_max = np.max(self.obc_2d, axis=0) # 2
        
        self.size = (self.max_dim - self.min_dim) + 5 # DIM
        
        # set start, end, obstacle, agent
        


        





