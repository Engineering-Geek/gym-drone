from __future__ import annotations

from typing import Set, Tuple, Type, Optional

import numpy as np
from gymnasium import Space
from gymnasium.core import ObsType, ActType, Env
from gymnasium.spaces import Dict as SpaceDict
from mujoco import MjModel, MjData, Renderer, mj_step, viewer, mj_resetData
from ray.rllib.env.multi_agent_env import MultiAgentEnv, AgentID
from ray.rllib.utils.typing import MultiEnvDict, MultiAgentDict

from gym_drone.drones.core.base_drone import BaseDrone, BaseTarget
from gym_drone.utils.model_generator import get_model


class MultiAgentBaseDroneEnvironment(MultiAgentEnv, Env):
    """
    A multi-agent environment for simulating drones and targets within a MuJoCo environment. This class orchestrates
    the interactions between multiple drones and targets, managing their states, actions, observations, and rewards
    throughout the simulation. It is designed to be flexible, allowing for the integration of various types of drones
    and targets.

    The environment provides a comprehensive setup for conducting drone-target interaction simulations, making it
    suitable for research in areas like multi-agent systems, reinforcement learning, and robotics.

    :param num_agents: The number of drone agents in the environment.
    :param DroneClass: The class of the drone to be used, which should inherit from BaseDrone.
    :param num_targets: The number of targets in the environment.
    :param spacing: The spacing between drones or targets.
    :param world_bounds: The boundaries of the world where agents can operate.
    :param respawn_box: The box within which drones can be respawned.
    :param spawn_angles: The range of angles for drones' initial orientations.
    :param calculate_drone_to_drone: Flag indicating whether to calculate relational data between drones.
    :param calculate_drone_to_target: Flag indicating whether to calculate relational data between drones and targets.
    :param render_mode: The mode used for rendering the environment (e.g., 'free', 'drone_pov', or None).
    :param kwargs: Additional keyword arguments for drone initialization.

    Usage:
        - Initialize the environment with the desired number of agents, drone class, and target configurations.
        - Use the step method to advance the simulation by one timestep, passing the actions for each agent.
        - Access the observations, rewards, done and info signals to interact with the environment.

    Note:
        This class is intended to be extended and customized. Users can create subclasses to modify the behavior
        and characteristics of the environment, such as custom collision handling, observation space definition,
        and reward computation.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, num_agents: int, DroneClass: Type[BaseDrone], num_targets: int, spacing: float,
                 world_bounds: np.ndarray, respawn_box: np.ndarray, spawn_angles: np.ndarray,
                 calculate_drone_to_drone: bool, calculate_drone_to_target: bool, render_mode: str,
                 n_phi: int = None, n_theta: int = None, ray_max_distance: float = None,
                 **kwargs):
        """
        Initializes the multi-agent drone environment with the specified configuration.

        :param num_agents: The number of drone agents to be included in the environment.
        :param DroneClass: The drone class, derived from BaseDrone, to instantiate for each agent.
        :param num_targets: The number of targets to include in the environment.
        :param spacing: The spacing between each agent or target within the environment.
        :param world_bounds: A numpy array specifying the boundaries of the environment.
        :param respawn_box: A numpy array defining the volume in which drones can be respawned at the beginning of an episode or after being reset.
        :param spawn_angles: A numpy array specifying the range of angles for the initial orientation of the drones.
        :param calculate_drone_to_drone: A boolean indicating whether relational data between drones should be calculated.
        :param calculate_drone_to_target: A boolean indicating whether relational data between drones and targets should be calculated.
        :param render_mode: A string specifying the rendering mode for the environment. Choose from 'free', 'drone_pov', or None.

        :keyword light_height: (float) The height at which the environment's light source is placed.
        :keyword drone_height: (float) The height at which drones are initialized or respawned.
        :keyword target_dimensions_range: (numpy.ndarray) The range of dimensions for targets within the environment.
        :keyword timestep: (float) The timestep used for the simulation.
        :keyword noise: (float) The standard deviation of the noise applied to the drone's positional data.
        :keyword bullet_velocity: (float) The velocity at which drones' bullets are shot.
        :keyword target_color: (numpy.ndarray) The default color for targets in the environment.
        :keyword frame_skip: (int) The number of simulation steps to skip for each environment step.
        :keyword fps: (int) The frames per second at which the environment is rendered.
        """
        # region Model Initialization
        mj_model = get_model(
            num_agents=num_agents,
            num_targets=num_targets,
            map_bounds=world_bounds,
            light_height=kwargs.get("light_height", 5),
            spacing=spacing,
            drone_height=kwargs.get("drone_height", 0.5),
            target_dimensions_range=kwargs.get("target_dimensions_range",
                                               np.array([[0.05, 0.05, 0.05], [0.2, 0.2, 0.2]])),
        )
        mj_model.opt.timestep = kwargs.get("timestep", 0.01)
        
        self.model: MjModel = mj_model
        self.data: MjData = MjData(self.model)
        self.render_mode = render_mode
        assert render_mode in ["free", "drone_pov", None], \
            "Invalid render mode. Choose from 'free', 'drone_pov', or None."
        self.handler: viewer.Handle = (
            viewer.launch_passive(self.model, self.data)) if render_mode in ["free", "drone_pov"] else None
        self.calculate_drone_to_drone = calculate_drone_to_drone
        self.calculate_drone_to_target = calculate_drone_to_target
        self.world_bounds = world_bounds if world_bounds is not None else np.array([[-25, -25, 0], [25, 25, 10]])
        self.max_distance = np.linalg.norm(self.world_bounds[1] - self.world_bounds[0])
        self.noise = kwargs.get("noise", 0.01)
        
        self.num_agents = num_agents
        self.agent_ids: Set[AgentID] = set(range(num_agents))
        self.target_ids = set(range(num_targets))
        # endregion
        
        # region Relational Data Initialization
        self.drone_positions = np.zeros((num_agents, 3))
        self.target_positions = np.zeros((num_targets, 3))
        self.drone_to_drone_deltas = np.zeros((num_agents, num_agents, 3)) if calculate_drone_to_drone else None
        self.drone_to_target_delta = np.zeros((num_agents, num_agents, 3)) if calculate_drone_to_target else None
        
        self.drone_to_drone_distance = np.zeros((num_agents, num_agents)) if calculate_drone_to_drone else None
        self.drone_to_drone_sin_theta = np.zeros((num_agents, num_agents)) if calculate_drone_to_drone else None
        self.drone_to_drone_cos_theta = np.zeros((num_agents, num_agents)) if calculate_drone_to_drone else None
        self.drone_to_drone_theta = np.zeros((num_agents, num_agents)) if calculate_drone_to_drone else None
        self.drone_to_drone_cartesian = np.zeros((num_agents, num_agents, 3)) if calculate_drone_to_drone else None
        
        self.drone_to_target_distance = np.zeros((num_agents, num_targets)) if calculate_drone_to_target else None
        self.drone_to_target_sin_theta = np.zeros((num_agents, num_targets)) if calculate_drone_to_target else None
        self.drone_to_target_cos_theta = np.zeros((num_agents, num_targets)) if calculate_drone_to_target else None
        self.drone_to_target_theta = np.zeros((num_agents, num_targets)) if calculate_drone_to_target else None
        self.drone_to_target_cartesian = np.zeros((num_agents, num_targets, 3)) if calculate_drone_to_target else None
        # endregion
        
        # region BaseDrone and BaseTarget Initialization
        self.drones = {agent_id: DroneClass(
            model=self.model,
            data=self.data,
            agent_id=agent_id,
            spawn_box=respawn_box,
            spawn_angles=spawn_angles,
            bullet_velocity=kwargs.get("bullet_velocity", 3.0),
            drone_to_drone_distance=self.drone_to_drone_distance[agent_id] if calculate_drone_to_drone else None,
            drone_to_target_distance=self.drone_to_target_distance[agent_id] if calculate_drone_to_target else None,
            drone_to_drone_sin_theta=self.drone_to_drone_sin_theta[agent_id] if calculate_drone_to_drone else None,
            drone_to_target_sin_theta=self.drone_to_target_sin_theta[agent_id] if calculate_drone_to_target else None,
            drone_to_drone_cos_theta=self.drone_to_drone_cos_theta[agent_id] if calculate_drone_to_drone else None,
            drone_to_target_cos_theta=self.drone_to_target_cos_theta[agent_id] if calculate_drone_to_target else None,
            drone_to_drone_theta=self.drone_to_drone_theta[agent_id] if calculate_drone_to_drone else None,
            drone_to_target_theta=self.drone_to_target_theta[agent_id] if calculate_drone_to_target else None,
            drone_to_drone_cartesian=self.drone_to_drone_cartesian[agent_id] if calculate_drone_to_drone else None,
            drone_to_target_cartesian=self.drone_to_target_cartesian[agent_id] if calculate_drone_to_target else None,
            max_world_diagonal=np.linalg.norm(self.world_bounds[1] - self.world_bounds[0]),
        ) for agent_id in self.agent_ids}
        self.targets = {target_id: BaseTarget(
            model=self.model,
            data=self.data,
            target_id=target_id,
            spawn_box=np.array([
                [self.world_bounds[0][0], self.world_bounds[0][1], 0.1],
                [self.world_bounds[1][0], self.world_bounds[1][1], 10]
            ]),
            spawn_angles=np.array([[0, 0, 0], [2 * np.pi, 2 * np.pi, 2 * np.pi]]),
            target_color=kwargs.get("target_color", np.array([0, 1, 0, 1]))
        ) for target_id in self.target_ids}
        self.drone_body_ids = [drone.body_id for drone in self.drones.values()]
        # endregion
        
        MultiAgentEnv.__init__(self)
        
        # region Pre-computation for collision checks
        self._bullet_geom_ids_to_drones = {
            drone.bullet.geom_id: drone for drone in self.drones.values()
        }
        self._drone_geom_ids_to_drones = {
            geom_id: drone for drone in self.drones.values() for geom_id in drone.geom_ids
        }
        self._target_geom_ids_to_targets = {
            target.geom_id: target for target in self.targets.values()
        }
        # endregion
        
        # region Floor Geometry
        self.floor_geom = self.model.geom("floor")
        self.floor_geom_id = self.floor_geom.id
        # endregion
        
        # region Pre-computation for rendering
        self.frame_skip = kwargs.get("frame_skip", 1)
        self.fps = kwargs.get("fps", 60)
        self.render_every = max(1, int((self.fps * self.frame_skip) / self.fps))
        self.physics_step_index = 0
        # endregion
        
        if n_phi and n_theta and ray_max_distance:
            self.init_rays(n_phi, n_theta, ray_max_distance)
        elif sum([n_phi, n_theta, ray_max_distance]) > 0:
            raise ValueError("All of n_phi, n_theta, and ray_max_distance must be specified.")
        
        # region Space Initialization
        self.observation_space: SpaceDict[AgentID, Space[ObsType]] = SpaceDict({
            agent_id: drone.observation_space for agent_id, drone in self.drones.items()
        })
        self.action_space: SpaceDict[AgentID, Space[ObsType]] = SpaceDict({
            agent_id: drone.action_space for agent_id, drone in self.drones.items()
        })
        # endregion
    
    def init_rays(self, n_theta: int = 16, n_phi: int = 8, max_distance: float = 10.0) -> None:
        """
        Initializes the rays for each drone.
        """
        for drone in self.drones.values():
            drone.init_rays(
                n_theta=n_theta,
                n_phi=n_phi,
                ray_max_distance=max_distance
            )
    
    def init_cameras(self, camera_type: str, renderer: Renderer, depth: bool = False) -> None:
        """
        Initializes the cameras for each drone.
        """
        for drone in self.drones.values():
            drone.init_camera(
                camera_type=camera_type,
                renderer=renderer,
                depth=depth
            )
    
    def collisions(self) -> None:
        """
        Checks and processes collisions between drones, bullets, targets, and the environment.
        """
        for contact in self.data.contact:
            geom_1_id, geom_2_id = contact.geom1, contact.geom2
            
            # region Collision Object Retrieval
            drone_bullet_1 = self._bullet_geom_ids_to_drones.get(geom_1_id)
            drone_1 = self._drone_geom_ids_to_drones.get(geom_1_id)
            target_1 = self._target_geom_ids_to_targets.get(geom_1_id)
            
            drone_bullet_2 = self._bullet_geom_ids_to_drones.get(geom_2_id)
            drone_2 = self._drone_geom_ids_to_drones.get(geom_2_id)
            target_2 = self._target_geom_ids_to_targets.get(geom_2_id)
            # endregion
            
            # region Bullet to Drone Collision
            if drone_bullet_1 and drone_2 and drone_bullet_1 != drone_2:
                drone_bullet_1.shot_drone = True
                drone_2.got_shot = True
                continue
            if drone_bullet_2 and drone_1 and drone_bullet_2 != drone_1:
                drone_bullet_2.shot_drone = True
                drone_1.got_shot = True
                continue
            # endregion
            
            # region Drone to Floor Collision
            if drone_1 and geom_2_id == self.floor_geom_id:
                drone_1.hit_floor = True
                continue
            if drone_2 and geom_1_id == self.floor_geom_id:
                drone_2.hit_floor = True
                continue
            # endregion
            
            # region Bullet to Target Collision
            if drone_bullet_1 and target_2:
                drone_bullet_1.shot_target = True
                continue
            if drone_bullet_2 and target_1:
                drone_bullet_2.shot_target = True
                continue
            # endregion
            
            # region Drone to Target Collision
            if drone_1 and target_2:
                drone_1.crash_target = True
                continue
            if drone_2 and target_1:
                drone_2.crash_target = True
                continue
            # endregion
            
            # region Drone to Drone Collision
            if drone_1 and drone_2 and drone_1 != drone_2:
                drone_1.crash_drone = True
                drone_2.crash_drone = True
                continue
            # endregion
    
    def act(self, action_dict: ActType) -> None:
        """
        Executes actions for each drone based on the provided action dictionary.
        """
        for agent_id, action in action_dict.items():
            self.drones[agent_id].act(action)
    
    def reset_agents(self) -> None:
        """
        Respawns all drones at new locations within the spawn box.
        """
        for drone in self.drones.values():
            drone.respawn()
    
    def reset_agent_flags(self) -> None:
        """
        Resets status flags for all drones in the environment.
        """
        for drone in self.drones.values():
            drone.reset_default_flags()
    
    def update_drone_relations(self) -> None:
        """
        Updates the relational data between drones and between drones and targets. This method computes
        normalized distances, angles, sine and cosine values of angles, and Cartesian coordinates
        for each drone-drone and drone-target pair. These computed values are normalized with respect
        to the maximum distance in the environment and stored in corresponding class attributes.
    
        The method handles two types of relationships:
            1. BaseDrone-to-BaseDrone: Calculates relative positions and orientations between each pair of drones.
            2. BaseDrone-to-BaseTarget: Calculates relative positions and orientations between each drone and target.
    
        The calculations are as follows:
    
        For BaseDrone-to-BaseDrone relationships:
            - Distance: \(d_{ij} = \frac{\| \mathbf{p}_i - \mathbf{p}_j \|}{\text{max\_distance}}\)
            - Angle (Theta): \(\theta_{ij} = \text{atan2}(y_{ij}, x_{ij})\)
            - Sine of Theta: \(\sin(\theta_{ij})\)
            - Cosine of Theta: \(\cos(\theta_{ij})\)
            - Cartesian Coordinates: \(\Delta \mathbf{p}_{ij} = \frac{\mathbf{p}_i - \mathbf{p}_j}{\text{max\_distance}} + \mathcal{N}(0, \text{noise})\)
    
        For BaseDrone-to-BaseTarget relationships:
            - Distance: \(d_{ik} = \frac{\| \mathbf{p}_i - \mathbf{t}_k \|}{\text{max\_distance}}\)
            - Angle (Theta): \(\theta_{ik} = \text{atan2}(y_{ik}, x_{ik})\)
            - Sine of Theta: \(\sin(\theta_{ik})\)
            - Cosine of Theta: \(\cos(\theta_{ik})\)
            - Cartesian Coordinates: \(\Delta \mathbf{p}_{ik} = \frac{\mathbf{p}_i - \mathbf{t}_k}{\text{max\_distance}} + \mathcal{N}(0, \text{noise})\)
    
        Where:
            - \( \mathbf{p}_i \) and \( \mathbf{p}_j \) are the positions of drones i and j, respectively.
            - \( \mathbf{t}_k \) is the position of target k.
            - \(\mathcal{N}(0, \text{noise})\) represents Gaussian noise with mean 0 and standard deviation defined by the 'noise' attribute.
            - \(\text{max\_distance}\) is the maximum possible distance in the environment, used for normalization.
    
        Note:
            - The distances and Cartesian coordinates are normalized by the maximum possible distance in the environment to ensure they are within a [0, 1] range.
            - This function updates class attributes that store the computed relational data, which can then be used for downstream tasks like observation generation or reward calculation.
        """
        # Update drone positions
        for i, agent_id in enumerate(self.agent_ids):
            self.drone_positions[i, :] = self.data.xpos[self.drones[agent_id].body_id]
        
        # Calculate drone-to-drone relationships if enabled
        if self.calculate_drone_to_drone:
            # Calculate relative positions
            np.subtract(self.drone_positions[:, np.newaxis, :], self.drone_positions, out=self.drone_to_drone_deltas)
            
            # Compute distances and normalize
            distances = np.linalg.norm(self.drone_to_drone_deltas, axis=2)
            np.divide(distances, self.max_distance, out=self.drone_to_drone_distance)
            np.fill_diagonal(self.drone_to_drone_distance, 0)  # Avoid self-distance
            
            # Calculate angles and add noise
            noise = np.random.normal(0, self.noise, self.drone_to_drone_theta.shape)
            np.arctan2(self.drone_to_drone_deltas[..., 0], self.drone_to_drone_deltas[..., 1],
                       out=self.drone_to_drone_theta)
            self.drone_to_drone_theta -= np.pi / 2  # Rotate frame so that 0 points towards y-axis
            self.drone_to_drone_theta += noise  # Add noise
            self.drone_to_drone_theta = (self.drone_to_drone_theta + np.pi) % (2 * np.pi) - np.pi  # Normalize angles
            
            # Calculate sine and cosine for the angles
            np.sin(self.drone_to_drone_theta, out=self.drone_to_drone_sin_theta)
            np.cos(self.drone_to_drone_theta, out=self.drone_to_drone_cos_theta)
            
            # Update Cartesian coordinates
            np.divide(self.drone_to_drone_deltas, self.max_distance, out=self.drone_to_drone_cartesian)
            np.add(self.drone_to_drone_cartesian, np.random.normal(0, self.noise, self.drone_to_drone_deltas.shape),
                   out=self.drone_to_drone_cartesian)
        
        # Calculate drone-to-target relationships if enabled
        if self.calculate_drone_to_target:
            # Update target positions
            target_positions = np.array(
                [self.data.geom_xpos[self.targets[target_id].geom_id] for target_id in self.target_ids])
            
            # Calculate relative positions
            np.subtract(self.drone_positions[:, np.newaxis, :], target_positions, out=self.drone_to_target_delta)
            
            # Compute distances and normalize
            distances = np.linalg.norm(self.drone_to_target_delta, axis=2)
            np.divide(distances, self.max_distance, out=self.drone_to_target_distance)
            
            # Calculate angles and add noise
            noise = np.random.normal(0, self.noise, self.drone_to_target_theta.shape)
            np.arctan2(self.drone_to_target_delta[..., 0], self.drone_to_target_delta[..., 1],
                       out=self.drone_to_target_theta)
            self.drone_to_target_theta -= np.pi / 2  # Rotate frame so that 0 points towards y-axis
            self.drone_to_target_theta += noise  # Add noise and normalize
            self.drone_to_target_theta = (self.drone_to_target_theta + np.pi) % (2 * np.pi) - np.pi
            
            # Calculate sine and cosine for the angles
            np.sin(self.drone_to_target_theta, out=self.drone_to_target_sin_theta)
            np.cos(self.drone_to_target_theta, out=self.drone_to_target_cos_theta)
            
            # Update Cartesian coordinates
            np.divide(self.drone_to_target_delta, self.max_distance, out=self.drone_to_target_cartesian)
            np.add(self.drone_to_target_cartesian, np.random.normal(0, self.noise, self.drone_to_target_delta.shape),
                   out=self.drone_to_target_cartesian)
    
    def update_out_of_bounds(self) -> None:
        """
        Checks and updates the out-of-bounds status for each drone and bullet.
        """
        # Using broadcasting to compare all positions with bounds in one go
        drone_positions = np.vstack([self.data.xpos[drone.body_id][:3] for drone in self.drones.values()])
        bullet_positions = np.vstack([self.data.xpos[drone.bullet.body_id][:3] for drone in self.drones.values()])
        
        # Combining all bounds checks into single operations
        drone_out_of_bounds = np.any(
            (drone_positions < self.world_bounds[0]) | (drone_positions > self.world_bounds[1]), axis=1)
        bullet_out_of_bounds = np.any(
            (bullet_positions < self.world_bounds[0]) | (bullet_positions > self.world_bounds[1]), axis=1)
        
        # Update out-of-bounds status for drones and their bullets
        for drone, drone_oob, bullet_oob in zip(self.drones.values(), drone_out_of_bounds, bullet_out_of_bounds):
            drone.out_of_bounds = drone_oob
            drone.bullet_out_of_bounds = bullet_oob
    
    def update_drones(self, render: bool = False) -> None:
        """
        Updates the state of the drones based on the current environment.
        """
        for drone in self.drones.values():
            drone.update(render=render)
            if drone.done or drone.truncated:
                drone.respawn()
            if self.model.body_contype[drone.bullet.body_id] == 0:
                drone.bullet.reset()
            drone.custom_update()
    
    @property
    def step_results(self) -> Tuple[MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict]:
        """
        Compiles the results of the step function.
        """
        self.physics_step_index += 1
        return self.observation, self.reward, self.truncated, self.done, self.log_info
    
    def step(self, action_dict: ActType) -> Tuple[MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict]:
        """
        Advances the environment by one time step based on the actions of each agent.
        """
        self.reset_agent_flags()
        self.act(action_dict)
        mj_step(self.model, self.data, self.frame_skip)
        self.update_drone_relations()
        self.collisions()
        self.update_out_of_bounds()
        self.update_drones(render=self.physics_step_index % self.render_every == 0)
        return self.step_results
    
    def render(self) -> None:
        """
        Renders the current state of the environment.
        """
        
        if self.handler is not None:
            self.handler.sync()
    
    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """
        Resets the environment to its initial state.
        """
        mj_resetData(self.model, self.data)
        self.reset_agents()
        self.reset_agent_flags()
        self.update_drone_relations()
        return self.observation, self.reward
    
    def close(self):
        """
        Closes the environment and releases resources.
        """
        if self.handler is not None:
            self.handler.close()
    
    @property
    def observation(self) -> MultiAgentDict:
        """
        Retrieves observations for each drone in the environment.
        """
        return {agent_id: self.drones[agent_id].observation for agent_id in self.agent_ids}
    
    @property
    def reward(self) -> MultiAgentDict:
        """
        Computes the reward for each drone based on the current state.
        """
        return {agent_id: self.drones[agent_id].reward for agent_id in self.agent_ids}
    
    @property
    def log_info(self) -> MultiAgentDict:
        """
        Provides additional information about the environment's state.
        """
        return {agent_id: self.drones[agent_id].log_info for agent_id in self.agent_ids}
    
    @property
    def done(self) -> MultiAgentDict:
        """
        Determines whether the episode has completed for each agent.
        """
        return {agent_id: self.drones[agent_id].done for agent_id in self.agent_ids}
    
    @property
    def truncated(self) -> MultiAgentDict:
        """
        Checks if the episode is truncated for each agent.
        """
        return {agent_id: self.drones[agent_id].truncated for agent_id in self.agent_ids}
