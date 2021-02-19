import matplotlib.pyplot as plt
import numpy as np
from Dynamic_2D_env.entities import *

class Simple2D():
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
        

if __name__ == '__main__':
    simple2d = Simple2D()

    for i in range(10000):
        simple2d.visualize(i)



    # env = np.zeros([200, 200, 3], np.uint8)
    # start = Zone(100, 200, size = [20,20], sim= None)
    # end = Zone(100, 0, size=[20,20], sim= None)

    # obs1 = Obstacle(25, 15, size=[20,20], transform=[10, 0], sim= None)
    # obs2 = Obstacle(180, 100, size=[10, 10], transform=[-10,0], sim=None)

    # for i in range(100):
    #     plt.cla()      
    #     start.draw(env)
    #     end.draw(env)
    #     obs1.draw(env)
    #     obs2.draw(env)

    #     obs1.cycle()
    #     obs2.cycle()

    #     plt.imshow(env)
    #     plt.pause(0.01)