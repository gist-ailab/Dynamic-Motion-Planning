from pyrep import PyRep
from pyrep.objects.joint import Joint
from pyrep.objects.shape import Shape
from os import path
from os.path import join
import os
import random
import numpy as np

from pyrep_environments.obstacles import Obstacle3D

current_path = path.dirname(path.abspath(__file__))
scene_file = join(current_path, "scene", "complex_3D.ttt") 

class Complex3D:
    def __init__(self, headless=False):
        self._pr = PyRep()
        self._pr.launch(scene_file=scene_file, headless=headless)
        self._pr.start()

        self.workspace_base = Shape("workspace")
        self.workspace = self._get_worksapce()
        
        self.obstacles = []
        self.velocity_scale = 0
        self.repiration_cycle = 0

    def _get_worksapce(self):
        base_pos = self.workspace_base.get_position()
        bbox = self.workspace_base.get_bounding_box()
        min_pos = [bbox[2*i] + base_pos[i] for i in range(3)]
        max_pos = [bbox[2*i+1] + base_pos[i] for i in range(3)]
        return [min_pos, max_pos]

    def reset(self, obstacle_num, velocity_scale, respiration_cycle=0):
        for obstacle in self.obstacles:
            obstacle.remove()
        self._pr.step()
        
        self.velocity_scale = velocity_scale
        self.repiration_cycle = respiration_cycle

        self.obstacles = []
        for i in range(obstacle_num):
            obs = Obstacle3D.create_random_obstacle(workspace=self.workspace,
                                                  velocity_scale=velocity_scale,
                                                  respiration_cycle=respiration_cycle)
            self.obstacles.append(obs)
        self._pr.step()

    def step(self):
        # update config
        for obs in self.obstacles:
            if self.velocity_scale > 0:
                obs.keep_velocity()
            if self.repiration_cycle > 0:
                obs.respire()
        self._pr.step()
