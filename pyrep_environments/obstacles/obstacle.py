from pyrep.objects.shape import Shape
from pyrep.backend import sim
from pyrep.const import PrimitiveShape

class Obstacle:
    """Obstacle in CoppeliaSim Scene
    - created based on PyRep Shape object
    """
    def __init__(self, size, type: PrimitiveShape):
        self.size = size
        self.type = type
        
        self.obj = Shape.create(type, size=list(size))

        self.set_mass(1e+2) # 1e-9
        self.set_friction(0) # 0 ~ 1000
        self.set_restitution(10) # 0 ~ 10
        
    #region Dynamic property
    def set_velocity(self, velocity):
        handle = self.obj.get_handle()
        self._set_velocity(handle=handle, lin_velocity=velocity)
    def get_velocity(self):
        return self.obj.get_velocity()[0]
    def set_mass(self, mass: float):
        self.obj.set_mass(mass)
    def set_friction(self, friction: float):
        self.obj.set_bullet_friction(friction)
    def set_restitution(self, restitution: float):
        """set restitution for Object
            larger restitution => more elastic
        Args:
            restitution (float): [0 ~ 10]
        """
        handle = self.obj.get_handle()
        self._set_bullet_restitution(handle, restitution)
    def set_size_by_factor(self, scaling_factor):
        handle = self.obj.get_handle()
        self._set_object_size_by_factor(handle, scaling_factor)
    
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

    #endregion

    def remove(self):
        self.obj.remove()
    