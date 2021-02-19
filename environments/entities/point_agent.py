import numpy as np
import cv2

from .entity import Entity

class PointAgent(Entity):
    def __init__(self, x, y, color='g'):
        super(PointAgent, self).__init__(x,y,color)

    def set_pos(self, x, y):
        self.x = x
        self.y = y
    
    def draw(self, env_npy):
        cv2.circle(env_npy, [self.x, self.y], radian = 1, color = self.color, thickness=-1)