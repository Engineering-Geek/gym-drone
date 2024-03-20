import numpy as np

from gym_drone.drones.lidar_drones import ShootingLidarDrone
from gym_drone.environments.core.base_environment import MultiAgentBaseDroneEnvironment


class LidarDroneBattleRoyal(ShootingLidarDrone):
    def __init__(self, **kwargs):
        super(LidarDroneBattleRoyal, self).__init__(**kwargs)
        self.staying_alive_reward = kwargs.get("staying_alive_reward", 1.0)
        self.hit_reward = kwargs.get("hit_reward", 10.0)
        self.hit_penalty = kwargs.get("hit_penalty", 10.0)
        self.floor_penalty = kwargs.get("floor_penalty", 10.0)
        self.out_of_bounds_penalty = kwargs.get("out_of_bounds_penalty", 10.0)
        self.bullet_out_of_bounds_penalty = kwargs.get("bullet_out_of_bounds_penalty", 10.0)
        self.aim_reward = kwargs.get("aim_reward", 10.0)
        self.shooting_penalty = kwargs.get("shooting_penalty", .05)
    
    @property
    def aim_score(self):
        return np.nanmax(self.drone_to_drone_cos_theta)
    
    @property
    def reward(self) -> float:
        reward = 0
        
        # sparse rewards and penalties
        if self.hit_floor:
            reward -= self.floor_penalty
        if self.got_shot:
            reward -= self.hit_penalty
        if self.shot_drone:
            reward += self.hit_reward
            
        # continuous rewards and penalties
        reward += self.staying_alive_reward * self.model.opt.timestep
        reward += self.aim_score * self.aim_reward * self.model.opt.timestep
        reward -= self.shooting_penalty * self.just_shot
        
        return reward
    
    @property
    def log_info(self) -> dict:
        return {
            "hit_floor": self.hit_floor,
            "got_shot": self.got_shot,
            "shot_drone": self.shot_drone,
            "out_of_bounds": self.out_of_bounds,
            "bullet_out_of_bounds": self.bullet_out_of_bounds,
            "aim_score": self.aim_score
        }
    
    @property
    def done(self) -> bool:
        return self.hit_floor or self.got_shot
    
    @property
    def truncated(self) -> bool:
        return self.out_of_bounds
    
    def custom_update(self):
        pass
    
    def reset_custom_flags(self):
        pass


class LidarBattleRoyal(MultiAgentBaseDroneEnvironment):
    def __init__(self, num_agents: int = 5, spacing: float = 3.0,
                 world_bounds: np.ndarray = None, respawn_box: np.ndarray = None, spawn_angles: np.ndarray = None,
                 n_phi: int = 16, n_theta: int = 16, ray_max_distance: float = 10, render_mode: str = None,
                 **kwargs):
        
        if world_bounds is None:
            world_bounds = np.array([[-5, -5, 0.5], [5, 5, 5]])
        if respawn_box is None:
            respawn_box = np.array([[-5, -5, 0.5], [5, 5, 5]])
        if spawn_angles is None:
            spawn_angles = np.array([[0, 0, 0], [2 * np.pi, 2 * np.pi, 2 * np.pi]])
        
        super(LidarBattleRoyal, self).__init__(
            num_agents=num_agents,
            DroneClass=LidarDroneBattleRoyal,
            num_targets=0,
            spacing=spacing,
            world_bounds=world_bounds,
            respawn_box=respawn_box,
            spawn_angles=spawn_angles,
            calculate_drone_to_drone=True,
            calculate_drone_to_target=False,
            render_mode=render_mode,
            n_phi=n_phi,
            n_theta=n_theta,
            ray_max_distance=ray_max_distance,
            **kwargs
        )



