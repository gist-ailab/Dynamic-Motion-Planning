import matplotlib.pyplot as plt
import numpy as np
from .entities import Zone, Obstacle
from .environment import Environment

class Dynamic2D(Environment):
    def __init__(self):
        self.reset()

    def reset(self):
        self.size = [20, 20]
        min_x, max_x = 0, self.size[0]
        min_y, max_y = 0, self.size[1]


        self.start = Zone(np.random.randint(min_x, max_x), max_y, size=[5,5])
        self.end = Zone(np.random.randint(min_x, max_x), min_y, size=[5,5])

        self.obstacles = []
        for i in range(7):
            init_x = np.random.randint(min_x+2, max_x-2)
            init_y = np.random.randint(min_y+2, max_y-2)
            transform = (np.random.random(size=2)-0.5)*2 # --> transform range = [-1, 1) for x, y
            self.obstacles.append(Obstacle(init_x, init_y, size=[2,2], transform= transform))

    def step(self):
        plt.imshow(self.env)

    def visualize(self, t):
        plt.cla()
        env = np.zeros([*self.size, 3], np.uint8)
        self.start.draw(env)
        self.end.draw(env)
        for obs in self.obstacles:
            obs.draw(env)
            obs.cycle()
        plt.imshow(env)
        plt.pause(0.01)
        
    def get_random_config(self):
        """extract non collision config in workspace

        """
        config = None
        return config

    def get_obstacle_cloud(self): # used for MPNet input
        obstacle_point_cloud = np.zeros((1400, 2))
        return obstacle_point_cloud