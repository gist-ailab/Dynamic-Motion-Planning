import numpy as np
from abc import *

class Entity():
    def __init__(self, x, y, color='r'):
        self.x=x
        self.y=y

        if color == 'r':
            self.color = [0, 0, 255]
        elif color == 'g':
            self.color = [0, 255, 0]
        elif color == 'b':
            self.color = [255, 0, 0]
        else:
            self.color = [200, 200, 200]
                
    @abstractmethod
    def cycle(self):
        pass

    @abstractmethod
    def draw(self):
        pass
        