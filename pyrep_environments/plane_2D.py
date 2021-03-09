from pyrep import PyRep
from pyrep.objects.joint import Joint
from pyrep.objects.shape import Shape
from os import path
from os.path import join
import os
import random
import numpy as np

from pyrep_environments.obstacle import Obstacle

current_path = path.dirname(path.abspath(__file__))

plane_2d_scene = join(current_path, "scene", "plane_2D.ttt") 


class SingleJointRobot:
    def __init__(self, joint_name):
        self.joint = Joint(joint_name)
        _, interval = self.joint.get_joint_interval()
        self.config_space = [interval[0], interval[0]+interval[1]]

    def set_joint_position(self, position: float):
        self.joint.set_joint_position(position)

    def get_random_config(self):
        return np.random.uniform(self.config_space[0], self.config_space[1])

    def set_random_config(self):
        random_config = self.get_random_config()
        self.set_joint_position(random_config)

class Plane2D:
    def __init__(self, headless=False):
        self._pr = PyRep()
        self._pr.launch(scene_file=plane_2d_scene, headless=headless)
        self._pr.start()

        self.workspace_base = Shape("workspace")
        self.workspace = self._get_worksapce()
        # self.agent = SingleJointRobot(joint_name="joint")
        self.obstacles = []

    def _get_worksapce(self):
        base_pos = self.workspace_base.get_position()
        bbox = self.workspace_base.get_bounding_box()
        min_pos = [bbox[2*i] + base_pos[i] for i in range(3)]
        max_pos = [bbox[2*i+1] + base_pos[i] for i in range(3)]
        return [min_pos, max_pos]

    def reset(self, obstacle_num=3):
        for obstacle in self.obstacles:
            obstacle.remove()
        self._pr.step()
        
        self.obstacles = []
        for i in range(obstacle_num):
            obs = Obstacle.create_random_obstacle(self.workspace, cycle=10)
            self.obstacles.append(obs)
        
    def step(self):
        # update config
        for obs in self.obstacles:
            obs.keep_velocity()
            obs.update_shape()
        self._pr.step()


    def check_collision(self):
        is_collision = False
        return is_collision

    


