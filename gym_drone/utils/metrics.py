from gym_drone.environments.core.base_environment import MultiAgentBaseDroneEnvironment
import time
from tqdm import tqdm


def simulation_rate(env: MultiAgentBaseDroneEnvironment, n_steps: int = 1000) -> float:
    """
    Calculate the simulation rate of the environment (seconds passed in simulation per second in real life).
    :param env: MultiAgentBaseDroneEnvironment
    :param n_steps: int (default: 1000)
    :return: float
    """
    start_time = time.time()
    for _ in tqdm(range(n_steps)):
        env.step(env.action_space.sample())
    end_time = time.time()
    return env.data.time / (end_time - start_time)


def simulation_time(env: MultiAgentBaseDroneEnvironment, n_steps: int = 1000) -> float:
    """
    Calculate the time taken to run the simulation for n_steps.
    :param env: MultiAgentBaseDroneEnvironment
    :param n_steps: int (default: 1000)
    :return: float
    """
    start_time = time.time()
    for _ in tqdm(range(n_steps)):
        env.step(env.action_space.sample())
    end_time = time.time()
    return end_time - start_time
    


