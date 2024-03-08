from typing import SupportsFloat, Any, Tuple

import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Space, Box, Sequence, Dict
from numpy.random import default_rng, Generator
from mujoco import MjModel, MjData


class CustomMujocoEnv(MujocoEnv):
    def __init__(self, xml_path, **kwargs):
        super().__init__(model_path=xml_path, frame_skip=kwargs.get('frame_skip', 1))
        self.rng = default_rng()
        self.model: MjModel  # Declare the model attribute
        self.data: MjData  # Declare the data attribute

        self.observation_space: Space = Dict({
            "drone_position": Box(low=-np.inf, high=np.inf, shape=(3,)),
            "drone_velocity": Box(low=-np.inf, high=np.inf, shape=(3,)),
            "drone_orientation": Box(low=-np.inf, high=np.inf, shape=(4,)),
            "drone_angular_velocity": Box(low=-np.inf, high=np.inf, shape=(3,)),
            "target_vector": Box(low=-np.inf, high=np.inf, shape=(3,)),
            "image_0": Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8),
            "image_1": Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8),
        })
        # Drone
        self.drone_spawn_x_range = kwargs.get('drone_spawn_x_range', (-10.0, 10.0))
        self.drone_spawn_y_range = kwargs.get('drone_spawn_y_range', (-10.0, 10.0))
        self.drone_spawn_z_range = kwargs.get('drone_spawn_z_range', (0.5, 3.0))
        self.drone_spawn_roll_range = kwargs.get('drone_spawn_roll_range', (-np.pi/6, np.pi/6))
        self.drone_spawn_pitch_range = kwargs.get('drone_spawn_pitch_range', (-np.pi/6, np.pi/6))
        self.drone_spawn_yaw_range = kwargs.get('drone_spawn_yaw_range', (-np.pi, np.pi))
        self.drone_spawn_max_velocity = kwargs.get('drone_spawn_max_velocity', 0.5)
        self.drone_spawn_max_angular_velocity = kwargs.get('drone_spawn_max_angular_velocity', 0.5)
        self.action_space: Space = Box(low=-1.0, high=1.0, shape=(self.model.nu,))

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        reward = self.compute_reward()
        done, truncated = self.check_done()
        info = self.compute_metrics()

        return observation, reward, done, truncated, info

    def _get_obs(self) -> ObsType:
        # Implement your observation logic here
        return np.concatenate([
            self.data.qpos.flat,  # Joint positions
            self.data.qvel.flat,  # Joint velocities
        ])

    def compute_reward(self) -> SupportsFloat:
        # Implement your reward logic here
        return 0.0

    def check_done(self) -> Tuple[bool, bool]:
        # Implement your logic to check if the episode is done
        return False

    def compute_metrics(self) -> dict[str, Any]:
        # Implement any additional metrics you want to track
        return {}

    def reset_model(self):
        # Required by MujocoEnv, called by reset()
        qpos = self.init_qpos + self.rng.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.rng.random(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        # Adjust the camera settings if needed
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def get_camera_images(self) -> Tuple[np.ndarray, np.ndarray]:
        pass
