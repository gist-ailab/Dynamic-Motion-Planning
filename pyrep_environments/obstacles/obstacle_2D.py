from pyrep.const import PrimitiveShape
from pyrep_environments.obstacles import Obstacle
import random
import numpy as np
import copy

class Obstacle2D(Obstacle):
    SHAPE_CANDIDATE=[PrimitiveShape.CUBOID,
                     PrimitiveShape.CYLINDER,
                     ]
    def __init__(self, size, type: PrimitiveShape, workspace,
                 velocity_scale=0, respiration_cycle=0):
        super(Obstacle2D, self).__init__(size, type)
        
        self.workspace = workspace # min pos, max pos
        
        self.velocity_scale = velocity_scale
        
        self.respiration_cycle = respiration_cycle
        self.respiration_count = 0
        self._initialize_respiration_ratio()
        
        self._initialize_position()
        self._initialize_velocity()
        
    def _initialize_respiration_ratio(self):
        x_ratio = np.random.uniform(1,3)
        y_ratio = np.random.uniform(1,3)
        
        x_step_ratio = self._calculate_step_ratio(x_ratio, self.respiration_cycle)
        y_step_ratio = self._calculate_step_ratio(y_ratio, self.respiration_cycle)
        self.upsize_ratio = np.array([x_step_ratio -1 , y_step_ratio-1, 0])

        x_step_ratio = self._calculate_step_ratio(1/x_ratio, self.respiration_cycle)
        y_step_ratio = self._calculate_step_ratio(1/y_ratio, self.respiration_cycle)
        self.downsize_ratio = np.array([1- x_step_ratio , 1 - y_step_ratio, 0])

    def _initialize_position(self):
        position = self.get_random_position()
        self.obj.set_position(list(position))

    def _initialize_velocity(self):
        direction = np.random.rand(3) - 0.5
        direction[2] = 0
        direction = direction / np.linalg.norm(direction)
        self.initial_velocity = self.velocity_scale * direction
        self.set_velocity(self.initial_velocity)

    def respire(self):
        """Update Obstacle shape
            - scale
        """
        assert self.respiration_cycle > 0, "respiration cycle set to 0"
        
        if (self.respiration_count // self.respiration_cycle) % 2 == 0:
            scaling_factor = 1 + self.upsize_ratio
        else:
            scaling_factor = 1 - self.downsize_ratio
        
        self.respiration_count += 1
        self.set_size_by_factor(scaling_factor)

    def keep_velocity(self):
        assert self.velocity_scale > 0, "velocity scale is set to 0"
        current_velocity, current_ang = self.get_velocity()
        current_scale = np.linalg.norm(np.array(current_velocity))
        if sum(current_velocity) == 0:
            self.set_velocity(self.initial_velocity)
        elif current_scale < self.velocity_scale:
            velocity = list(np.array(current_velocity) * self.velocity_scale / current_scale)
            self.set_velocity(velocity, current_ang)
        else:
            pass

    def get_random_position(self):
        min_pos = np.array(self.workspace[0]) + np.array(self.size)
        max_pos = np.array(self.workspace[1]) - np.array(self.size)
        z_pos = self.workspace[1][2]
        random_position = np.random.uniform(min_pos, max_pos)
        random_position[2] = z_pos

        return random_position

    @staticmethod
    def create_random_obstacle(workspace, velocity_scale=0, respiration_cycle=0):
        min_pos = np.array(workspace[0])
        max_pos = np.array(workspace[1])
        z_pos = (max_pos[2] + min_pos[2]) / 2
        z_size = max_pos[2] - min_pos[2]

        min_size = (max_pos - min_pos) / 10
        max_size = (max_pos - min_pos) / 5

        obs_workspace = copy.deepcopy(workspace)
        obs_workspace[0][2] = z_pos
        obs_workspace[1][2] = z_pos

        size = np.random.uniform(min_size, max_size)
        size[2] = z_size

        type = random.choice(Obstacle2D.SHAPE_CANDIDATE)

        return Obstacle2D(size, type, workspace, velocity_scale, respiration_cycle)

