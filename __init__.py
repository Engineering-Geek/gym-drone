from gymnasium.envs.registration import register
from gym_drone.environments.battle_royal import LidarBattleRoyal

register(
    id="LidarBattleRoyal-v0",
    entry_point="gym_drone.environments:LidarBattleRoyal",
    kwargs={
        "num_agents": 5,
        "spacing": 3.0,
        "world_bounds": None,
        "respawn_box": None,
        "spawn_angles": None,
        "n_phi": 16,
        "n_theta": 16,
        "ray_max_distance": 10,
    },
)
