
class Environment(object):
    """Base class of Motion Planning Environments
    """
    def __init__(self):
        pass    
    def reset(self):
        pass
    def step(self):
        pass
    def visualize(self):
        pass
    def get_random_config(self):
        pass
    