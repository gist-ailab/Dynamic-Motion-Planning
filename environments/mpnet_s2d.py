import matplotlib.pyplot as plt
import numpy as np
# from .entities import Zone, Obstacle, PointAgent
from .environment import Environment

class Point():
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

class MPNetSimple2D(Environment):
    def __init__(self, obc_file):
        self.obc = np.fromfile(obc_file) # 2800
        self.obc_2d = np.reshape(self.obc, (-1, 2)) # 1400, 2
        
        # set workspace
       
        # set start, end, obstacle, agent
        
        obc_min = np.min(self.obc_2d, axis=0) - 2
        
        obc_max = np.max(self.obc_2d, axis=0) + 2
        self.workspace = [obc_min, obc_max]
        
        self.obstacles_xy = []
        self.obstacles_clouds = np.reshape(self.obc_2d, (7, -1, 2))
        for i in range(7):
            obstacle_cloud = self.obstacles_clouds[i] # 200, 2
            x, y = np.mean(obstacle_cloud, axis=0)
            self.obstacles_xy.append((x, y))
        

        r_point = self.get_random_point()
        self.start = Point(r_point[0], r_point[1])
        r_point = self.get_random_point()
        self.end = Point(r_point[0], r_point[1])
        
        self.path = np.array([[self.start.x, self.start.y]])

    def visualize(self, next_config):
        plt.cla()
        # draw start and end
        plt.scatter(self.start.x, self.start.y, c="g")
        plt.scatter(self.end.x, self.end.y, c="g")

        # draw agent
        self.path = np.append(self.path, [next_config], axis=0)
        plt.scatter(self.path[:, 0], self.path[:, 1], c="r")

        # draw obstacle map
        plt.scatter(self.obc_2d[:,0], self.obc_2d[:,1], c="b")
        
        plt.show(block=False)
        plt.pause(0.01)
    
    def get_random_point(self):
        val = np.random.uniform(self.workspace[0], self.workspace[1])
        while self.collision_check(val):
            val = np.random.uniform(self.workspace[0], self.workspace[1])
        return val

    def collision_check(self, point):
        is_collision = False
        for obs_xy in self.obstacles_xy:
            for i in range(2):
                dist = abs(obs_xy[i] - point[i])
                if dist < 5 / 2.0:
                    is_collision = True
                    break

            if is_collision:
                return is_collision
        return is_collision




    def get_obstacle_cloud(self):
        return self.obc

    def get_start_and_end(self):
        return [self.start.x, self.start.y], [self.end.x, self.end.y]
        





