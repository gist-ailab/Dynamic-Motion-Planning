import matplotlib.pyplot as plt
import numpy as np
import cv2

from entities import *


if __name__ == '__main__':

    start = Zone(100, 200, size = [20,20], sim= None)
    end = Zone(100, 0, size=[20,20], sim= None)

    obs1 = Obstacle(25, 15, size=[20,20], transform=[10, 0], sim= None)
    obs2 = Obstacle(180, 100, size=[10, 10], transform=[-10,0], sim=None)

    for i in range(100):
        plt.cla()
        env = np.zeros([200, 200, 3], np.uint8)

        start.draw(env)
        end.draw(env)
        obs1.draw(env)
        obs2.draw(env)

        obs1.cycle()
        obs2.cycle()

        plt.imshow(env)
        plt.pause(0.01)