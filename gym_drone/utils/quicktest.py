from mujoco import viewer
from gym_drone.utils.model_generator import get_model
import numpy as np


m = get_model(
    num_agents=2,
    num_targets=2,
    light_height=1.0,
    spacing=1.0,
    drone_height=0.5,
    target_dimensions_range=np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]),
    map_bounds=np.array([[-5, -5, 0.5], [5, 5, 5]])
)

viewer.launch(m)


