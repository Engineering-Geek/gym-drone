from abc import ABC

import numpy as np
from gymnasium import Space
from gymnasium.spaces import Box, Tuple as TupleSpace, Discrete
from ray.rllib.utils.typing import EnvObsType

from gym_drone.drones.core.base_drone import BaseDrone


class LidarBaseDrone(BaseDrone, ABC):
    def __init__(self, **kwargs):
        super(LidarBaseDrone, self).__init__(**kwargs)
        self.return_gyro_data = np.zeros(3)
        self.return_accel_data = np.zeros(3)
        self.return_position = np.zeros(3)
        self.lidar_data = None
    
    @property
    def observation_space(self) -> Space[EnvObsType]:
        sensor_space = Box(
            low=-1, high=1, shape=(6,), dtype=np.float32
        )
        position_space = Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
        )
        lidar_space = Box(
            low=0, high=1, shape=(len(self.distances),), dtype=np.float32
        )
        return TupleSpace((sensor_space, position_space, lidar_space))
    
    def observation(self) -> EnvObsType:
        if not self.lidar_data:
            self.lidar_data = np.zeros(len(self.distances))
            
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
            np.clip(
                self.distances,
                a_min=0, a_max=1, out=self.lidar_data
            )
        return self.return_accel_data, self.return_gyro_data, self.return_position, self.lidar_data
    
    @property
    def action_space(self) -> Space:
        sample_motor_id = self.actuator_ids[0]
        motor_space = Box(
            low=0, high=self.model.actuator_ctrlrange[sample_motor_id, 1], shape=(4,), dtype=np.float32
        )
        return motor_space
    
    def act(self, action: np.ndarray) -> None:
        self.data.ctrl[self.actuator_ids] = action


class ShootingLidarDrone(LidarBaseDrone, ABC):
    def __init__(self, **kwargs):
        super(ShootingLidarDrone, self).__init__(**kwargs)
        
    @property
    def action_space(self) -> Space:
        sample_motor_id = self.actuator_ids[0]
        motor_space = Box(
            low=0, high=self.model.actuator_ctrlrange[sample_motor_id, 1], shape=(4,), dtype=np.float32
        )
        bullet_space = Discrete(2)
        return TupleSpace((motor_space, bullet_space))
    
    def act(self, action: np.ndarray) -> None:
        self.data.ctrl[self.actuator_ids] = action[0]
        if action[1]:
            self.bullet.shoot()
            self.just_shot = True
        
        