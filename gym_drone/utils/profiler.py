import cProfile
from tqdm import tqdm
from gym_drone.environments.battle_royal import LidarBattleRoyal
import numpy as np
from argparse import ArgumentParser


def create_env(render: bool, n_agents: int = 5, n_phi: int = 16, n_theta: int = 16,
               ray_max_distance: int = 10) -> LidarBattleRoyal:
    spawn_box = np.array([[-5, -5, 0.5], [5, 5, 5]])
    world_bounds = np.array([[-10, -10, -0.1], [10, 10, 5]])
    env = LidarBattleRoyal(render_mode="human" if render else None, world_bounds=world_bounds, respawn_box=spawn_box,
                           num_agents=n_agents, n_phi=n_phi, n_theta=n_theta, ray_max_distance=10)
    return env


def profile(n_steps: int = 1000, n_agents: int = 5, n_phi: int = 16, n_theta: int = 16, max_distance: int = 10,
            render: bool = False, filename: str = "profile.prof"):
    env = create_env(render=render, n_agents=n_agents, n_phi=n_phi, n_theta=n_theta, ray_max_distance=max_distance)
    env.reset()
    profiler = cProfile.Profile()
    profiler.enable()
    for i in tqdm(range(n_steps), desc="Profiling"):
        action = env.action_space.sample()
        env.step(action)
        if render:
            env.render()
    profiler.disable()
    profiler.dump_stats(filename)
    env.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--n_agents", type=int, default=5)
    parser.add_argument("--n_phi", type=int, default=16)
    parser.add_argument("--n_theta", type=int, default=16)
    parser.add_argument("--max_distance", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--filename", type=str, default="profile.prof")
    args = parser.parse_args()
    profile(n_steps=args.n_steps, n_agents=args.n_agents, n_phi=args.n_phi, n_theta=args.n_theta,
            max_distance=args.max_distance, render=args.render, filename=args.filename)
    
    
