class Planner():
    def __init__(self):
        self.start_config = None
        self.goal_confgi = None
    
    def get_path(self):
        assert NotImplementedError, "Shoud be implemented in each Child"