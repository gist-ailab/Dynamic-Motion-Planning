from pyrep.objects.shape import Shape
from pyrep.backend import sim
from pyrep.const import PrimitiveShape

import random
import numpy as np

class Obstacle:
    def __init__(self, size, type: PrimitiveShape, velocity, workspace, respiration_cycle=10):
        self.size = size
        self.workspace = workspace # min pos, max pos
        self.respiration_cycle = respiration_cycle
        self.respiration_count = 0

        position = self.get_random_position()
        self.obj = Shape.create(type, size=list(size), position=list(position))

        self.obj.set_collidable(False)
        self.set_mass(1e+2) # 1e-9
        self.set_friction(0) # 0 ~ 1000
        self.set_restitution(10) # 0 ~ 10
        
        self.velocity_scale = np.linalg.norm(np.array(velocity))
        self.set_velocity(velocity)

    def update(self):
        """Update Obstacle config
            - keep velocity scale
            - scale
        """
        current_velocity = self.get_velocity()
        current_scale = np.linalg.norm(np.array(current_velocity))
        if current_scale < self.velocity_scale:
            velocity = list(np.array(current_velocity) * self.velocity_scale / current_scale)
            self.set_velocity(velocity)

        if (self.respiration_count // self.respiration_cycle) % 2 == 0:
            scaling_factor = [1.001] * 3
        else:
            scaling_factor = [0.999] * 3
        self.respiration_count += 1
        self.set_size_by_factor(scaling_factor)
    
    def get_random_position(self):
        min_pos = np.array(self.workspace[0]) + np.array(self.size)
        max_pos = np.array(self.workspace[1]) - np.array(self.size)
        random_position = np.random.uniform(min_pos, max_pos)

        return random_position

    def set_velocity(self, velocity):
        handle = self.obj.get_handle()
        self._set_velocity(handle=handle, lin_velocity=velocity)
    def get_velocity(self):
        return self.obj.get_velocity()[0]

    def set_mass(self, mass):
        self.obj.set_mass(mass)
    def set_friction(self, friction):
        self.obj.set_bullet_friction(friction)
    def set_restitution(self, restitution):
        handle = self.obj.get_handle()
        self._set_bullet_restitution(handle, restitution)
    def set_size_by_factor(self, scaling_factor):
        # size_factor: 0.5 ~ 1.5
        handle = self.obj.get_handle()
        self._set_object_size_by_factor(handle, scaling_factor)
        
    def remove(self):
        self.obj.remove()

    @staticmethod
    def _set_velocity(handle, lin_velocity=[0, 0, 0], ang_velocity=[0, 0, 0]):
        # reset dynamic object
        sim.simResetDynamicObject(handle)
        
        # shapefloatparam_init_velocity_x
        sim.simSetObjectFloatParameter(handle, sim.sim_shapefloatparam_init_velocity_x, lin_velocity[0])
        # shapefloatparam_init_velocity_y 
        sim.simSetObjectFloatParameter(handle, sim.sim_shapefloatparam_init_velocity_y, lin_velocity[1])
        # shapefloatparam_init_velocity_z  
        sim.simSetObjectFloatParameter(handle, sim.sim_shapefloatparam_init_velocity_z, lin_velocity[2])

        # shapefloatparam_init_velocity_a
        sim.simSetObjectFloatParameter(handle, sim.sim_shapefloatparam_init_velocity_a, ang_velocity[0])
        # shapefloatparam_init_velocity_b
        sim.simSetObjectFloatParameter(handle, sim.sim_shapefloatparam_init_velocity_b, ang_velocity[1])
        # shapefloatparam_init_velocity_g
        sim.simSetObjectFloatParameter(handle, sim.sim_shapefloatparam_init_velocity_g, ang_velocity[2])
    @staticmethod
    def _set_bullet_restitution(handle, restitution):
        sim.simSetEngineFloatParameter(handle, sim.sim_bullet_body_restitution, restitution)
    @staticmethod
    def _set_object_size_by_factor(handle, scaling_factor):
        # sim.simSetObjectFloatParameter(handle, sim.sim_objfloatparam_size_factor, size_factor)
        sim.lib.simScaleObject(handle, scaling_factor[0], scaling_factor[1], scaling_factor[2], 0)
    @staticmethod
    def create_random_obstacle(workspace, cycle):
        min_pos = np.array(workspace[0])
        max_pos = np.array(workspace[1])
        
        min_size = (max_pos - min_pos) / 10
        max_size = (max_pos - min_pos) / 5

        size = np.random.uniform(min_size, max_size)
        velocity = np.random.uniform(max_size)
        type = random.choice(list(PrimitiveShape))
        # type = PrimitiveShape.SPHERE

        return Obstacle(size, type, velocity, workspace, respiration_cycle=cycle)

