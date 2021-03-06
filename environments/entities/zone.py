import numpy as np
import cv2

from .entity import Entity

class Zone(Entity):
    def __init__(self, x, y, size, color='b'):
        super(Zone, self).__init__(x,y,color)

        self.size = size

        self.size_x = size[0]/2
        self.size_y = size[1]/2


    def cycle(self):
        pass

    def draw(self, env_npy):
        pt_top = (int(self.x-self.size_x), int(self.y-self.size_y))
        pt_bottom = (int(self.x+self.size_x), int(self.y+self.size_y))

        cv2.rectangle(env_npy, pt_top, pt_bottom, self.color, thickness=-1)