import matplotlib.pyplot as plt
import numpy as np
from .entities import Zone, Obstacle, PointAgent
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
        self.start = Zone(size=[1,1],x=0,y=0)
        self.end = Zone(size=[1,1],x=0,y=0)
        self.agent = PointAgent(size=[1,1],x=0,y=0)
        self.obstacles = []
        self.obstacles_clouds = np.reshape(self.obc_2d, (7, -1, 2))
        for i in range(7):
            obstacle_cloud = self.obstacles_clouds[i] # 200, 2
            x, y = np.mean(obstacle_cloud, axis=0)
            self.obstacles.append(Obstacle(x, y, size=[2,2], transform= [0, 0]))
        
    def visualize(self):
        plt.cla()
        env = np.zeros([*self.size, 3], np.uint8)
        self.start.draw(env)
        self.end.draw(env)
        for obs in self.obstacles:
            obs.draw(env)
            obs.cycle()
        self.agent.draw(env)
        plt.imshow(env)
        plt.pause(0.01)
    
    def get_obstacle_cloud(self):
        return self.obc_2d


        





