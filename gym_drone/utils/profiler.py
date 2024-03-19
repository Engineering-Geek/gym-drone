from tqdm import tqdm

from gym_drone.battle_royal import BattleRoyalDroneEnvironment
import cProfile
import pstats


def simulate(env: BattleRoyalDroneEnvironment):
    for _ in tqdm(range(1000)):
        env.step(env.action_space.sample())
    

if __name__ == "__main__":
    environment = BattleRoyalDroneEnvironment(
        num_agents=5,
        num_targets=2,
        light_height=1.0,
        spacing=1.0,
        drone_height=0.5,
        target_dimensions_range=[[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]],
        map_bounds=[[-5, -5, 0.5], [5, 5, 5]],
        time_limit=100.0,
        alive_reward=1.0,
        hit_target_reward=10.0,
        crash_into_target_penalty=10.0,
        shot_drone_reward=10.0,
        floor_penalty=10.0,
        out_of_bounds_penalty=10.0,
        facing_target_reward=10.0,
        facing_drone_reward=10.0,
        n_theta=10,
        n_phi=10,
    )
    environment.reset()
    profiler = cProfile.Profile()
    profiler.enable()
    simulate(environment)
    profiler.disable()
    environment.close()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.dump_stats('profile_results.prof')
    
    
