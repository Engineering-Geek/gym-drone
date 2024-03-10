from _base_drone_env import _BaseDroneEnv


class SuicideDroneEnv(_BaseDroneEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tolerance_distance = 0.05  # meters
        self.max_time = 20  # seconds
        

