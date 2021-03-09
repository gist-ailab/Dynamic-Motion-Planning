import numpy as np
import cv2

from enum import Enum
from .entity import Entity

class EntityShape(Enum):
    RECTANGLE = 0
    CIRCLE = 1
    ELLIPSE = 3
    POLYGON = 4
    
class Obstacle(Entity):
    def __init__(self, x, y, size, transform, color='r'):
        super(Obstacle, self).__init__(x,y,color)

        self.transform = transform
        self.size = size

        self.size_x = int(size[0]/2)
        self.size_y = int(size[1]/2)

    def cycle(self):
        self.x += self.transform[0]
        self.y += self.transform[1]

    def draw(self, env_npy):
        pt_top = (int(self.x-self.size_x), int(self.y-self.size_y))
        pt_bottom = (int(self.x+self.size_x), int(self.y+self.size_y))

        # check obstacle is out from map
        hit = False
        if pt_top[0]<0 or pt_top[1]<0:
            hit = True
        if pt_bottom[0]>=len(env_npy[0]) or pt_bottom[1]>=len(env_npy[1]):
            hit = True

        # if obstacle is out from map, inverse transform
        if hit:
            self.transform = [-self.transform[0], -self.transform[1]]
        cv2.circle
        
        cv2.rectangle(env_npy, pt_top, pt_bottom, self.color, thickness=-1)

