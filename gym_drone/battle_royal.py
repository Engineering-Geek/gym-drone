import numpy as np
from gymnasium import Space
from gymnasium.spaces import Box, Tuple as TupleSpace, MultiDiscrete
from ray.rllib.utils.typing import EnvObsType

from gym_drone.core import Drone, MultiAgentBaseDroneEnvironment
from gym_drone.utils.model_generator import get_model


class BattleRoyalDrone(Drone):
    def __init__(self, **kwargs):
        super(BattleRoyalDrone, self).__init__(depth=False, **kwargs)
        # ---------------------------------------------------------
        self.time_limit = float(kwargs.get("time_limit", 100.0))  # seconds
        # ---------------------------------------------------------
        self.passed_time_limit = False
        # ---------------------------------------------------------
        self.alive_reward = float(kwargs.get("alive_reward", 1.0))  # Reward for staying alive per second
        self.hit_target_reward = float(kwargs.get("hit_target_reward", 10.0))
        self.crash_into_target_penalty = float(kwargs.get("crash_into_target_penalty", 10.0))
        self.shot_drone_reward = float(kwargs.get("shot_drone_reward", 10.0))
        self.floor_penalty = float(kwargs.get("floor_penalty", 10.0))
        self.out_of_bounds_penalty = float(kwargs.get("out_of_bounds_penalty", 10.0))
        self.facing_target_reward = float(kwargs.get("facing_target_reward", 10.0))
        self.facing_drone_reward = float(kwargs.get("facing_drone_reward", 10.0))
        # ---------------------------------------------------------
        self.return_gyro_data = np.zeros(3)
        self.return_accel_data = np.zeros(3)
        self.return_position = np.zeros(3)
        self.lidar_data = np.zeros(len(self.distances))
    
    @property
    def done(self) -> bool:
        return self.data.time >= self.time_limit
    
    @property
    def truncated(self) -> bool:
        return self.data.time >= self.time_limit
    
    @property
    def reward(self) -> float:
        reward = self.alive_reward * self.model.opt.timestep        # Reward for staying alive per second
        if self.hit_floor:
            reward -= self.floor_penalty
        if self.shot_target:
            reward += self.hit_target_reward
        if self.crash_target:
            reward -= self.crash_into_target_penalty
        if self.shot_drone:
            reward += self.shot_drone_reward
        if self.out_of_bounds:
            reward -= self.out_of_bounds_penalty
        
        return reward
    
    @property
    def log_info(self) -> dict:
        return {
            "hit_floor": self.hit_floor,
            "shot_target": self.shot_target,
            "crash_target": self.crash_target,
            "shot_drone": self.shot_drone,
            "out_of_bounds": self.out_of_bounds
        }
    
    @property
    def observation_space(self) -> Space[EnvObsType]:
        """
        Returns the observation space for the environment.
            - Drone's accelerometer and gyroscope data (between -1 and 1, normalized using sensor cutoffs)
            - Normalized position of the drone in the world (between -1 and 1, normalized using world bounds)
            - LiDAR data (rays, between 0 and 1, normalized using max distance)
        :return: gym.Space
        """
        # Accelerometer and Gyroscope data (between -1 and 1, normalized using sensor cutoffs)
        sensor_space = Box(
            low=-1, high=1, shape=(6,), dtype=np.float32
        )
        # Normalized position of the drone in the world (between -1 and 1, normalized using world bounds)
        position_space = Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
        )
        # LiDAR data (rays, between 0 and 1, normalized using max distance)
        lidar_space = Box(
            low=0, high=1, shape=(len(self.distances),), dtype=np.float32
        )
        return TupleSpace((sensor_space, position_space, lidar_space))
    
    def observe(self, render: bool) -> EnvObsType:
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
                self.distances / self.max_world_diagonal,
                a_min=0, a_max=1, out=self.lidar_data
            )
        return self.return_accel_data, self.return_gyro_data, self.return_position, self.lidar_data
    
    @property
    def action_space(self) -> Space:
        # 4 motors (between 0 and motor_max_ctrl), 1 binary shoot action
        sample_motor_id = self.actuator_ids[0]
        motor_space = Box(
            low=0, high=self.model.actuator_ctrlrange[sample_motor_id, 1], shape=(4,), dtype=np.float32
        )
        shoot_space = MultiDiscrete(1)
        return TupleSpace((motor_space, shoot_space))
    
    def act(self, action: int) -> None:
        motor_action, shoot_action = action
        self.data.ctrl[self.actuator_ids] = motor_action
        if shoot_action == 1:
            self.bullet.shoot()
    
    def reset_flags(self):
        self.passed_time_limit = False
        

class BattleRoyalDroneEnvironment(MultiAgentBaseDroneEnvironment):
    def __init__(self, num_agents: int, num_targets: int = 2, **kwargs):
        super().__init__(
            num_agents=num_agents,
            DroneClass=BattleRoyalDrone,
            num_targets=num_targets,
            world_bounds=np.array([
                [-50, -50, -0.01],
                [50, 50, 20]
            ]),
            respawn_box=np.array([
                [-0.5, -0.5, 0.1],
                [0.5, 0.5, 1]
            ]),
            spawn_angles=np.array([
                [0, 0, 0],
                [2 * np.pi, 2 * np.pi, 2 * np.pi]
            ]),
            calculate_drone_to_drone=False,
            calculate_drone_to_target=False,
            depth=False,
            camera_type="stereo",
            fps=15,
            frame_skip=10,
            enable_ray=True,
            **kwargs
        )
