from typing import SupportsFloat, Any, Tuple

import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Space, Box, Sequence, Dict
from numpy.random import default_rng, Generator
from mujoco import MjModel, MjData


class CustomMujocoEnv(MujocoEnv):

    def __init__(self, xml_path, sim_rate, dt, **kwargs):
        height = kwargs.get('height', 480)
        width = kwargs.get('width', 640)
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / dt)),
        }

        observation_space: Space[ObsType] = Dict({
            "drone_position": Box(low=-np.inf, high=np.inf, shape=(3,)),
            "drone_velocity": Box(low=-np.inf, high=np.inf, shape=(3,)),
            "drone_orientation": Box(low=-np.inf, high=np.inf, shape=(4,)),
            "drone_angular_velocity": Box(low=-np.inf, high=np.inf, shape=(3,)),
            "target_vector": Box(low=-np.inf, high=np.inf, shape=(3,)),
            "image_0": Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8),
            "image_1": Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8),
        })

        MujocoEnv.__init__(
            self,
            model_path=xml_path,
            frame_skip=sim_rate,
            observation_space=observation_space,
            width=width,
            height=height,
            **kwargs,
        )
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        self.rng = default_rng()
        self.model: MjModel  # Declare the model attribute
        self.data: MjData  # Declare the data attribute
        print(self.mujoco_renderer.viewer)
        # Drone
        self.drone_spawn_x_range = kwargs.get('drone_spawn_x_range', (-10.0, 10.0))
        self.drone_spawn_y_range = kwargs.get('drone_spawn_y_range', (-10.0, 10.0))
        self.drone_spawn_z_range = kwargs.get('drone_spawn_z_range', (0.5, 3.0))
        self.drone_spawn_roll_range = kwargs.get('drone_spawn_roll_range', (-np.pi / 6, np.pi / 6))
        self.drone_spawn_pitch_range = kwargs.get('drone_spawn_pitch_range', (-np.pi / 6, np.pi / 6))
        self.drone_spawn_yaw_range = kwargs.get('drone_spawn_yaw_range', (-np.pi, np.pi))
        self.drone_spawn_max_velocity = kwargs.get('drone_spawn_max_velocity', 0.5)
        self.drone_spawn_max_angular_velocity = kwargs.get('drone_spawn_max_angular_velocity', 0.5)
        self.action_space: Space = Box(low=-1.0, high=1.0, shape=(self.model.nu,))

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.pre_simulation()
        self.do_simulation(action, self.frame_skip)
        return self.observation, self.reward, self.truncated, self.done, self.metrics

    def pre_simulation(self) -> None:
        pass

    @property
    def observation(self) -> ObsType:
        # Implement your observation logic here
        return np.concatenate([
            self.data.qpos.flat,  # Joint positions
            self.data.qvel.flat,  # Joint velocities
        ])

    @property
    def reward(self) -> SupportsFloat:
        raise NotImplementedError

    @property
    def done(self) -> bool:
        raise NotImplementedError

    @property
    def truncated(self) -> bool:
        raise NotImplementedError

    @property
    def metrics(self) -> dict[str, Any]:
        raise NotImplementedError

    def reset_model(self):
        # Required by MujocoEnv, called by reset()
        qpos = self.init_qpos + self.rng.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.rng.random(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self.observation()

    def viewer_setup(self):
        # Adjust the camera settings if needed
        self.render()

    def get_camera_images(self) -> Tuple[np.ndarray, np.ndarray]:
        pass


if __name__ == "__main__":
    env = CustomMujocoEnv(xml_path="/home/nmelgiri/PycharmProjects/gym-drone/gym_drone/assets/UAV/scene.xml", sim_rate=1, dt=0.01)
