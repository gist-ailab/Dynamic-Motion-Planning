import matplotlib.pyplot as plt
import numpy as np
# from .entities import Zone, Obstacle, PointAgent
from .environment import Environment

class MPNetSimple2D(Environment):
    def __init__(self, obc_file, path_file=None):
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
        
        self.true_path = None
        if path_file is not None:
            true_path = np.fromfile(path_file)
            self.true_path = true_path.reshape(-1, 2).astype(np.float32)
            self.start = self.true_path[0]
            self.end = self.true_path[-1]
        else:
            self.start = self.get_random_point()
            self.end = self.get_random_point()
            
        self.path = np.array([[self.start[0], self.start[1]]])

    def visualize(self, next_config d):
        plt.cla()
        # draw start and end
        plt.scatter(self.start[0], self.start[1], c="g")
        plt.scatter(self.end[0], self.end[1], c="g")

        # draw agent
        self.path = np.append(self.path, [next_config], axis=0)
        plt.scatter(self.path[:, 0], self.path[:, 1], c="r")

        # draw obstacle map
        plt.scatter(self.obc_2d[:,0], self.obc_2d[:,1], c="b")
        
        plt.show(block=False)
        plt.pause(0.01)
    
    def set_decoder_view(self, decoder_view):
        self.dobc_2d = decoder_view
        
    def visualize_with_decodedview(self, next_config):        
        plt.cla()
        plt.scatter(self.start[0], self.start[1], c="g")
        plt.scatter(self.end[0], self.end[1], c="g")

        if self.true_path is not None:
            plt.scatter(self.true_path[1:-1, 0], self.true_path[1:-1, 1], c="c")

        # draw agent
        self.path = np.append(self.path, [next_config], axis=0)
        plt.scatter(self.path[1:, 0], self.path[1:, 1], c="r")

        # draw obstacle map
        plt.scatter(self.obc_2d[:,0], self.obc_2d[:,1], c="b")

        # draw decoder_view of obstacle map
        plt.scatter(self.dobc_2d[:,0], self.dobc_2d[:,1], c ='y')
        
        plt.show(block=False)
        # plt.show()
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
        return [self.start[0], self.start[1]], [self.end[0], self.end[1]]
        
