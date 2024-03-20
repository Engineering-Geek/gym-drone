from abc import ABC

import numpy as np
from gymnasium import Space
from gymnasium.spaces import Box, Tuple as TupleSpace, Discrete
from ray.rllib.utils.typing import EnvObsType

from gym_drone.drones.core.base_drone import BaseDrone


class CameraBaseDrone(BaseDrone, ABC):
    
    def __init__(self, **kwargs):
        super(CameraBaseDrone, self).__init__(**kwargs)
        self.return_gyro_data = np.zeros(3)
        self.return_accel_data = np.zeros(3)
        self.return_position = np.zeros(3)
        self.image_normalizing_factor: float = None
        self.image_1 = None
        self.image_2 = None
    
    @property
    def observation_space(self) -> Space[EnvObsType]:
        if self.image_normalizing_factor is None:
            self.image_normalizing_factor = self.ray_max_distance if self.depth else 255
        if self.camera_type == "stereo":
            self.image_1 = np.zeros(self._image_1.shape)
            self.image_2 = np.zeros(self._image_1.shape)
        elif self.camera_type == "mono":
            self.image_1 = np.zeros(self._image_1.shape)
        sensor_space = Box(
            low=-1, high=1, shape=(6,), dtype=np.float32
        )
        position_space = Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
        )
        dimensions = self._image_1.shape
        camera_1 = Box(
            low=0, high=1, shape=dimensions, dtype=np.float32
        )
        if self.camera_type == "stereo":
            camera_2 = Box(
                low=0, high=1, shape=dimensions, dtype=np.float32
            )
            return TupleSpace((sensor_space, position_space, camera_1, camera_2))
        return TupleSpace((sensor_space, position_space, camera_1))
    
    def observation(self) -> EnvObsType:
        if self.model.sensor_cutoff[self.gyro_id] and self.model.sensor_cutoff[self.accel_id]:
            np.clip(
                self.data.sensordata[self.gyro_id: self.gyro_id + 3] / self.model.sensor_cutoff[self.gyro_id],
                a_min=-1, a_max=1, out=self.return_gyro_data
            )
            np.clip(
                self.data.sensordata[self.accel_id: self.accel_id + 3] / self.model.sensor_cutoff[self.accel_id],
                a_min=-1, a_max=1, out=self.return_accel_data
            )
            np.clip(
                self.data.xpos[self.body_id] / self.max_world_diagonal,
                a_min=-1, a_max=1, out=self.return_position
            )
            if self.camera_type == "mono":
                np.clip(
                    self._image_1 / self.image_normalizing_factor,
                    a_min=0, a_max=1, out=self.image_1
                )
                return self.return_accel_data, self.return_gyro_data, self.return_position, self.image_1
            elif self.camera_type == "stereo":
                np.clip(
                    self._image_1 / self.image_normalizing_factor,
                    a_min=0, a_max=1, out=self.image_1
                )
                np.clip(
                    self._image_2 / self.image_normalizing_factor,
                    a_min=0, a_max=1, out=self.image_2
                )
                return self.return_accel_data, self.return_gyro_data, self.return_position, self.image_1, self.image_2
            else:
                raise ValueError("Camera type not supported, please use 'mono' or 'stereo' cameras.")
    
    @property
    def action_space(self) -> Space:
        sample_motor_id = self.actuator_ids[0]
        motor_space = Box(
            low=0, high=self.model.actuator_ctrlrange[sample_motor_id, 1], shape=(4,), dtype=np.float32
        )
        return motor_space
    
    def act(self, action: np.ndarray) -> None:
        self.data.ctrl[self.actuator_ids] = action


class ShootingCameraDrone(CameraBaseDrone, ABC):
    def __init__(self, **kwargs):
        super(ShootingCameraDrone, self).__init__(**kwargs)
    
    @property
    def action_space(self) -> Space:
        sample_motor_id = self.actuator_ids[0]
        motor_space = Box(
            low=0, high=self.model.actuator_ctrlrange[sample_motor_id, 1], shape=(4,), dtype=np.float32
        )
        bullet_space = Discrete(1)
        return TupleSpace((motor_space, bullet_space))
    
    def act(self, action: np.ndarray) -> None:
        self.data.ctrl[self.actuator_ids] = action[0]
        if action[1]:
            self.model.shoot(self.model, self.data, self.model.opt.timestep)

