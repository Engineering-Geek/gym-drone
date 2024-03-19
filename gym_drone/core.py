"""
This module provides a simulation environment for drones and targets using MuJoCo, where drones can interact with
targets and each other in a defined space. It defines classes for bullets, targets, drones, and the simulation
environment itself.

Classes:
    Bullet: Represents a bullet in the simulation, with methods to manage its shooting and resetting behavior.
    Target: Represents a target in the simulation, providing methods to reset its position and orientation.
    Drone: Defines a drone in the simulation, encapsulating its properties, actions, and interactions.
    BaseEnvironment: A multi-agent environment for drones and targets, managing the simulation state and agent interactions.

Bullet:
    The Bullet class encapsulates the properties and behaviors of bullets fired by drones in the environment. It manages
    bullet positioning, velocity, and resetting its state within the simulation.

Target:
    The Target class represents target objects within the simulation. Targets can be repositioned and reset, providing
    interactive elements for drones.

Drone:
    The Drone class represents an individual drone within the simulation. It includes properties like the drone's model,
    sensors, camera, and actions. Drones can observe their environment, take actions, and interact with targets and other
    drones.

BaseEnvironment:
    The BaseEnvironment class provides a multi-agent simulation environment where drones and targets can interact.
    It handles the simulation steps, agent actions, observations, and rewards, providing a comprehensive framework
    for drone-target interaction simulations.

The module is structured to be used with reinforcement learning algorithms, particularly in scenarios where multiple
drones interact within a shared environment, targeting objectives or engaging in simulated combat scenarios.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Set, Tuple, Type, Union, Optional

import numpy as np
import quaternion
from scipy.spatial.transform import Rotation as R
from gymnasium import Space
from gymnasium.core import ObsType, ActType, Env
from gymnasium.spaces import Dict as SpaceDict
from mujoco import MjModel, MjData, Renderer, mj_step, mj_name2id, mjtObj, viewer, mj_resetData, mj_multiRay, mjMAXVAL
from mujoco._structs import _MjModelGeomViews, _MjModelSensorViews
from ray.rllib.env.multi_agent_env import MultiAgentEnv, AgentID
from ray.rllib.utils.typing import EnvObsType, EnvActionType, MultiEnvDict, MultiAgentDict

from gym_drone.utils.model_generator import get_model


class Bullet:
    """
    Represents a bullet object in the simulation environment, managing its properties and interactions.

    Attributes:
        agent_id (AgentID): Identifier of the agent controlling this bullet.
        model (MjModel): The MuJoCo model associated with the simulation.
        data (MjData): The data structure containing the current state of the simulation.
        bullet_id (int): Unique identifier for the bullet within the simulation.
        bullet_velocity (float): The velocity at which the bullet is shot.
        parent (Drone): The drone object that fired this bullet.
        body_id (int): The ID of the bullet's body within the MuJoCo model.
        geom_id (int): The ID of the bullet's geometry within the MuJoCo model.
        spawn_site_id (int): The ID of the site where the bullet is spawned.
        joint_id (int): The ID of the joint associated with the bullet.
        bullet_color (np.ndarray): The RGBA color of the bullet.
        qpos_offset (int): Pre-computed offset for the bullet's position in the qpos array.
        qvel_offset (int): Pre-computed offset for the bullet's velocity in the qvel array.

    Methods:
        __init__(self, agent_id: AgentID, model: MjModel, data: MjData, bullet_id: int,
                 bullet_velocity: float, parent: Drone, **kwargs):
            Initializes a new instance of the Bullet class.
        
        shoot(self):
            Simulates the shooting action of the bullet.

        reset(self):
            Resets the bullet's state within the simulation.
    """
    
    def __init__(self, agent_id: AgentID, model: MjModel, data: MjData, bullet_id: int, bullet_velocity: float,
                 parent: Drone, **kwargs):
        """
        Initializes a new instance of the Bullet class.
        
        Arguments:
            :param agent_id: Identifier of the agent controlling this bullet.
            :param model: The MuJoCo model associated with the simulation.
            :param data: The data structure containing the current state of the simulation.
            :param bullet_id: Unique identifier for the bullet within the simulation.
            :param bullet_velocity: The velocity at which the bullet is shot.
            :param parent: The drone object that fired this bullet.
            :param kwargs: Additional keyword arguments.
            :keyword bullet_color: The RGBA color of the bullet.
        """
        self.agent_id = agent_id
        self.model = model
        self.data = data
        self.bullet_id = bullet_id
        self.bullet_velocity = bullet_velocity
        self.parent = parent
        
        bullet_name = f"bullet_{bullet_id}"
        self.body_id = mj_name2id(model, mjtObj.mjOBJ_BODY, bullet_name)
        self.geom_id = mj_name2id(model, mjtObj.mjOBJ_GEOM, bullet_name)
        self.spawn_site_id = mj_name2id(model, mjtObj.mjOBJ_SITE, f"bullet_spawn_{bullet_id}")
        self.joint_id = mj_name2id(model, mjtObj.mjOBJ_JOINT, f"bullet_joint_{bullet_id}")
        self.bullet_color = kwargs.get("bullet_color", np.array([1, 0, 0, 1]))
        
        # Pre-compute offsets
        self.qpos_offset = self.joint_id * 3
        self.qvel_offset = self.joint_id * 6
        
        # Set initial color (if color doesn't need to change dynamically)
        self.model.geom_rgba[self.geom_id] = self.bullet_color
    
    def shoot(self):
        """
        Simulates the shooting action of the bullet.
        """
        self.data.qpos[self.qpos_offset: self.qpos_offset + 3] = self.data.site_xpos[self.spawn_site_id]
        self.data.qvel[self.qvel_offset: self.qvel_offset + 3] = (self.parent.data.cvel[self.parent.body_id][:3] +
                                                                  [0, self.bullet_velocity, 0])
    
    def reset(self):
        """
        Resets the bullet's state within the simulation.
        """
        self.data.qpos[self.joint_id * 7: self.joint_id * 7 + 7] = [0, 0, 0, 1, 0, 0, 0]
        self.data.qvel[self.joint_id * 6: self.joint_id * 6 + 6] = [0, 0, 0, 0, 0, 0]
        self.data.geom_rgba[self.geom_id] = [0, 0, 0, 0]


class Target:
    """
    Represents a target object in the simulation environment, managing its properties and behavior.

    Attributes:
        model (MjModel): The MuJoCo model associated with the simulation.
        data (MjData): The data structure containing the current state of the simulation.
        target_id (int): Unique identifier for the target within the simulation.
        spawn_box (np.ndarray): The bounding box within which the target can be spawned.
        spawn_angles (np.ndarray): The range of angles for random orientation of the target.
        geom_id (int): The ID of the target's geometry within the MuJoCo model.
        target_color (np.ndarray): The RGBA color of the target.

    Methods:
        __init__(self, model: MjModel, data: MjData, target_id: int, spawn_box: np.ndarray,
                 spawn_angles: np.ndarray, target_color: np.ndarray = np.array([0, 1, 0, 1])):
            Initializes a new instance of the Target class.

        reset(self):
            Resets the target's position and orientation within the simulation.
    """
    
    def __init__(self, model: MjModel, data: MjData, target_id: int, spawn_box: np.ndarray,
                 spawn_angles: np.ndarray, target_color: np.ndarray = np.array([0, 1, 0, 1])):
        """
        Initializes a new instance of the Target class.

        Args:
            model (MjModel): The MuJoCo model associated with the simulation.
            data (MjData): The data structure containing the current state of the simulation.
            target_id (int): Unique identifier for the target within the simulation.
            spawn_box (np.ndarray): The bounding box within which the target can be spawned.
            spawn_angles (np.ndarray): The range of angles for random orientation of the target.
            target_color (np.ndarray, optional): The RGBA color of the target. Defaults to red.
        """
        self.model = model
        self.data = data
        self.target_id = target_id
        self.spawn_box = spawn_box
        self.spawn_angles = spawn_angles
        
        target_name = f"target_{target_id}"
        self.geom_id = mj_name2id(model, mjtObj.mjOBJ_GEOM, target_name)
        
        self.model.geom_rgba[self.geom_id] = target_color
    
    def reset(self):
        """
        Resets the target's position and orientation within the simulation based on the spawn box and angles.
        """
        self.data.geom_xpos[self.geom_id] = np.random.uniform(*self.spawn_box)
        self.data.geom_xquat[self.geom_id] = quaternion.from_euler_angles(
            np.random.uniform(*self.spawn_angles)).components


class Drone:
    """
    Represents a drone in the MuJoCo simulation environment, encapsulating its properties, sensors, and actions.
    
    Default Flags:
        - hit_floor: Indicates if the drone has hit the floor.
        - got_shot: Indicates if the drone has been hit by a bullet.
        - shot_target: Indicates if the drone has hit a target.
        - shot_drone: Indicates if the drone has shot another drone.
        - out_of_bounds: Indicates if the drone has gone out of bounds.
        - crash_target: Indicates if the drone has crashed into a target.
    
    Abstract Methods:
        - observation_space: Defines the observation space of the drone.
        - observe: Retrieves observations from the drone's sensors.
        - action_space: Defines the action space of the drone.
        - act: Executes an action taken by the drone.
        - reward: Calculates the reward for the drone's current state.
        - log_info: Provides additional information about the drone's state.
        - done: Determines whether the episode has completed for the drone.
        - truncated: Checks if the episode is truncated for the drone.
        - reset_flags: Resets custom flags or statuses of the drone.

    Attributes:
        model (MjModel): The MuJoCo model associated with the simulation.
        data (MjData): The data structure containing the current state of the simulation.
        agent_id (AgentID): Unique identifier for the drone within the simulation.
        renderer (Renderer): The renderer used for visualizing the simulation.
        camera_type (str): Specifies the type of camera ('stereo', 'mono', or None).
        depth (bool): Indicates if depth sensing is enabled.
        spawn_box (np.ndarray): Defines the volume in which the drone can be respawned.
        spawn_angles (np.ndarray): Specifies the range of angles for the drone's initial orientation.
        bullet_velocity (float): The velocity at which the drone's bullets are shot.
        _drone_to_drone_distance (np.ndarray): The distance between this drone and other drones.
        _drone_to_target_distance (np.ndarray): The distance between this drone and targets.
        _drone_to_drone_sin_theta (np.ndarray): The sine of the angle between this drone and other drones.
        _drone_to_target_sin_theta (np.ndarray): The sine of the angle between this drone and targets.
        _drone_to_drone_cos_theta (np.ndarray): The cosine of the angle between this drone and other drones.
        _drone_to_target_cos_theta (np.ndarray): The cosine of the angle between this drone and targets_.

    Methods:
        __init__: Initializes the Drone class with model, data, agent identifier, and other configurations.
        init_images: Initializes image arrays based on the camera type and depth sensing configuration.
        images: Returns the current image or images captured by the drone's camera(s).
        respawn: Respawns the drone at a new location and orientation within the spawn box.
        reset: Resets the drone's state, including its position, orientation, and bullet.
        observation_space: Abstract method to define the observation space of the drone.
        observe: Abstract method to obtain observations from the drone's sensors.
        action_space: Abstract method to define the action space of the drone.
        act: Abstract method for the drone to take actions within the environment.
        reward: Abstract method to calculate the reward for the drone's current state.
        log_info: Abstract method to log additional information about the drone's state.
        done: Abstract method to determine whether the drone's episode is complete.
        truncated: Abstract method to check if the drone's episode is truncated.
        reset_default_flags: Resets the default status flags of the drone.
        reset_flags: Abstract method to reset custom flags or statuses of the drone.
    """
    
    def __init__(
            self,
            model: MjModel,
            data: MjData,
            agent_id: AgentID,
            renderer: Renderer,
            camera_type: str,
            depth: bool,
            spawn_box: np.ndarray,
            spawn_angles: np.ndarray,
            bullet_velocity: float,
            drone_to_drone_distance: Optional[np.ndarray] = None,
            drone_to_target_distance: Optional[np.ndarray] = None,
            drone_to_drone_sin_theta: Optional[np.ndarray] = None,
            drone_to_target_sin_theta: Optional[np.ndarray] = None,
            drone_to_drone_cos_theta: Optional[np.ndarray] = None,
            drone_to_target_cos_theta: Optional[np.ndarray] = None,
            drone_to_drone_cartesian: Optional[np.ndarray] = None,
            drone_to_target_cartesian: Optional[np.ndarray] = None,
            n_phi: int = 36,
            n_theta: int = 36,
            ray_max_distance: float = 50,
            enable_ray: bool = False,
            max_world_diagonal: float = 100,
            **kwargs
    ):
        # region Basic Initialization
        self.model = model
        self.data = data
        self.agent_id = agent_id
        self.renderer = renderer
        self.camera_type = camera_type
        self.depth = depth
        self.spawn_box = spawn_box
        self.spawn_angles = spawn_angles
        self.bullet_velocity = bullet_velocity
        self.max_world_diagonal = max_world_diagonal
        # endregion
        
        # region Drone-to-Drone and Drone-to-Target Distances and Angles
        self._drone_to_drone_distance = drone_to_drone_distance
        self._drone_to_target_distance = drone_to_target_distance
        self._drone_to_drone_sin_theta = drone_to_drone_sin_theta
        self._drone_to_target_sin_theta = drone_to_target_sin_theta
        self._drone_to_drone_cos_theta = drone_to_drone_cos_theta
        self._drone_to_target_cos_theta = drone_to_target_cos_theta
        self._drone_to_drone_cartesian = drone_to_drone_cartesian
        self._drone_to_target_cartesian = drone_to_target_cartesian
        # endregion
        
        # region Ray Unit Vectors and Related Attributes
        self.enable_ray = enable_ray
        if enable_ray:
            self.initial_ray_unit_vectors = np.array(
                [
                    [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
                    for theta in np.linspace(0, np.pi, n_theta) for phi in np.linspace(0, 2 * np.pi, n_phi)
                ]
            )
            self.flattened_ray_unit_vectors = self.initial_ray_unit_vectors.copy().flatten()
            self.distances = np.zeros(n_phi * n_theta, dtype=np.float64)
            self.intersecting_geoms = np.zeros(n_phi * n_theta, dtype=np.int32)
            self.ray_max_distance = ray_max_distance
            self.num_rays = n_phi * n_theta
        # endregion
        
        # region Pre-Compute IDs
        drone_name = f"drone_{agent_id}"
        self.body_id = mj_name2id(model, mjtObj.mjOBJ_BODY, drone_name)
        self.geom_ids = [mj_name2id(model, mjtObj.mjOBJ_GEOM, f"{drone_name}_collision_{i}") for i in range(2)]
        self.free_joint_id = mj_name2id(model, mjtObj.mjOBJ_JOINT, f"{drone_name}_free_joint")
        self.free_joint_qpos_address = self.model.jnt_qposadr[self.free_joint_id]
        self.free_joint_qvel_address = self.model.jnt_dofadr[self.free_joint_id]
        # endregion
        
        # region Pre-Compute Camera IDs Based on Camera Type
        if self.camera_type == "mono":
            self.mono_camera_id = mj_name2id(model, mjtObj.mjOBJ_CAMERA, drone_name)
        elif self.camera_type == "stereo":
            self.left_camera_id = mj_name2id(model, mjtObj.mjOBJ_CAMERA, f"{drone_name}_left")
            self.right_camera_id = mj_name2id(model, mjtObj.mjOBJ_CAMERA, f"{drone_name}_right")
        # endregion
        
        # region Sensor IDs
        self.accel_id = mj_name2id(model, mjtObj.mjOBJ_SENSOR, f"accelerometer_{agent_id}")
        self.gyro_id = mj_name2id(model, mjtObj.mjOBJ_SENSOR, f"gyro_{agent_id}")
        # endregion
        
        # region Initialize Other Attributes
        self.initial_pos = self.data.xpos[self.body_id].copy()
        self.bullet = Bullet(agent_id, model, data, agent_id, bullet_velocity, self, **kwargs)
        self.front_left_actuator_id = mj_name2id(model, mjtObj.mjOBJ_ACTUATOR, f"front_left_{self.agent_id}")
        self.front_right_actuator_id = mj_name2id(model, mjtObj.mjOBJ_ACTUATOR, f"front_right_{self.agent_id}")
        self.back_left_actuator_id = mj_name2id(model, mjtObj.mjOBJ_ACTUATOR, f"back_left_{self.agent_id}")
        self.back_right_actuator_id = mj_name2id(model, mjtObj.mjOBJ_ACTUATOR, f"back_right_{self.agent_id}")
        self.actuator_ids = [
            self.front_left_actuator_id,
            self.front_right_actuator_id,
            self.back_left_actuator_id,
            self.back_right_actuator_id
        ]
        self._image_1: np.ndarray = None
        self._image_2: np.ndarray = None
        self.init_images()
        # endregion
        
        # region Flags for Various Conditions
        self.hit_floor = False
        self.got_shot = False
        self.shot_target = False
        self.shot_drone = False
        self.out_of_bounds = False
        self.crash_target = False
        # endregion
    
    @property
    def ray_distances(self) -> np.ndarray:
        """
        Returns the distances of the rays cast from the drone.

        Returns:
            np.ndarray: An array of distances of the rays cast from the drone.
        """
        return self.distances / self.ray_max_distance
    
    def init_images(self):
        """
        Initializes image arrays based on the camera configuration and whether depth sensing is enabled.
        """
        dimensions = (self.renderer.height, self.renderer.width, 3) if not self.depth else (
            self.renderer.height, self.renderer.width)
        self._image_1 = np.zeros(dimensions)
        if self.camera_type == "stereo":
            self._image_2 = np.zeros(dimensions)
    
    def images(self, render: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Returns the current image or images captured by the drone's camera(s).

        Returns:
            np.ndarray: An array or arrays representing the captured image(s).
        """
        if self.camera_type == "mono":
            if render:
                self.renderer.update_scene(self.data, camera=self.mono_camera_id)
                self.renderer.render(out=self._image_1)
            return self._image_1
        elif self.camera_type == "stereo":
            if render:
                self.renderer.update_scene(self.data, camera=self.left_camera_id)
                self.renderer.render(out=self._image_1)
                self.renderer.update_scene(self.data, camera=self.right_camera_id)
                self.renderer.render(out=self._image_2)
            return self._image_1, self._image_2
    
    def respawn(self):
        """
        Respawns the drone at a new location and orientation within the predefined spawn box.
        """
        pos = np.random.uniform(*self.spawn_box) + self.initial_pos[:3]
        quat = quaternion.from_euler_angles(np.random.uniform(*self.spawn_angles))
        self.data.qpos[self.free_joint_qpos_address: self.free_joint_qpos_address + 7] = np.concatenate(
            [pos, quaternion.as_float_array(quat)])
        self.data.qvel[self.free_joint_qvel_address: self.free_joint_qvel_address + 6].fill(0)
    
    def reset(self):
        """
        Resets the drone's state, including its position, orientation, and associated bullet.
        """
        self.bullet.reset()
        self.reset_flags()
        self.reset_default_flags()
        self.respawn()
    
    def reset_default_flags(self):
        """
        Resets the default status flags of the drone, including collision and status flags.
        """
        self.hit_floor = False
        self.got_shot = False
        self.shot_target = False
        self.shot_drone = False
        self.out_of_bounds = False
        self.crash_target = False
    
    @abstractmethod
    def observation_space(self) -> Space[EnvObsType]:
        """
        Defines the observation space of the drone. This method should be implemented by subclasses to
        specify how the drone perceives its environment.

        Returns:
            Space[EnvObsType]: The observation space of the drone.
        """
        raise NotImplementedError
    
    @abstractmethod
    def observe(self, render: bool) -> EnvObsType:
        """
        Retrieves observations from the drone's sensors. This method should be implemented by subclasses to
        provide sensor data or other relevant observations from the environment.

        Args:
            render (bool): Flag indicating whether to render the observation if applicable.

        Returns:
            EnvObsType: The observed data.
        """
        raise NotImplementedError
    
    @abstractmethod
    def action_space(self) -> Space[EnvActionType]:
        """
        Defines the action space of the drone. This method should be implemented by subclasses to specify
        the set of actions the drone can take.

        Returns:
            Space[EnvActionType]: The action space of the drone.
        """
        raise NotImplementedError
    
    @abstractmethod
    def act(self, action: EnvActionType):
        """
        Executes an action taken by the drone. This method should be implemented by subclasses to define how
        the drone responds to actions.

        Args:
            action (EnvActionType): The action to be executed.
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def reward(self) -> float:
        """
        Abstract method to calculate the reward for the drone's current state.

        Returns:
            float: The calculated reward.
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def log_info(self) -> dict:
        """
        Abstract method to log additional information about the drone's state.

        Returns:
            dict: A dictionary containing log information.
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def done(self) -> bool:
        """
        Abstract method to determine whether the drone's episode is complete.

        Returns:
            bool: True if the episode is complete, False otherwise.
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def truncated(self) -> bool:
        """
        Abstract method to check if the drone's episode is truncated.

        Returns:
            bool: True if the episode is truncated, False otherwise.
        """
        raise NotImplementedError
    
    @abstractmethod
    def reset_flags(self):
        """
        Abstract method to reset custom flags or statuses of the drone.
        """
        raise NotImplementedError
    
    def update_rays(self):
        if self.enable_ray:
            self.flattened_ray_unit_vectors[:] = R.from_quat(
                self.data.qpos[self.free_joint_qpos_address + 3: self.free_joint_qpos_address + 7]  # [x, y, z, w]
            ).apply(self.initial_ray_unit_vectors).flatten()  # [x, y, z] -> [x1, y1, z1, x2, y2, z2, ...]
            mj_multiRay(
                m=self.model,
                d=self.data,
                pnt=self.data.xpos[self.body_id],
                vec=self.flattened_ray_unit_vectors,
                geomgroup=None,
                flg_static=1,
                bodyexclude=self.body_id,
                geomid=self.intersecting_geoms,
                dist=self.distances,
                nray=self.num_rays,
                cutoff=mjMAXVAL
            )


class MultiAgentBaseDroneEnvironment(Env, MultiAgentEnv):
    """
    A multi-agent environment for simulating drones and targets in MuJoCo.

    Attributes:
        model (MjModel): The MuJoCo model associated with the environment.
        data (MjData): The data structure containing the current state of the simulation.
        renderer (Renderer): Renderer used for visualizing the simulation.
        camera_type (str): Specifies the type of camera ('stereo', 'mono', or None).
        depth (bool): Indicates if depth sensing is enabled.
        world_bounds (np.ndarray): Defines the boundaries of the world where agents can operate.
        drones (dict): A dictionary of drones present in the environment, keyed by agent IDs.
        targets (dict): A dictionary of targets present in the environment, keyed by target IDs.

    Methods:
        __init__: Initializes the environment with the specified number of agents, drone class, targets, and other configurations.
        collisions: Checks and processes collisions between drones, bullets, targets, and the environment.
        act: Executes actions for each drone based on the provided action dictionary.
        reset_agents: Respawns all drones at new locations within the spawn box.
        reset_agent_flags: Resets status flags for all drones in the environment.
        update_drone_relations: Updates the relational data between drones and between drones and targets.
        update_out_of_bounds: Checks and updates the out-of-bounds status for each drone.
        step_results: Compiles the results of the environment's step function.
        step: Advances the environment by one time step based on the actions of each agent.
        observation: Retrieves observations for each drone in the environment.
        reward: Computes the reward for each drone based on the current state.
        log_info: Provides additional information about the environment's state.
        done: Determines whether the episode has completed for each agent.
        truncated: Checks if the episode is truncated for each agent.
    """
    
    def __init__(self, num_agents: int, DroneClass: Type[Drone], num_targets: int, world_bounds: np.ndarray,
                 respawn_box: np.ndarray, spawn_angles: np.ndarray, camera_type: str, depth: bool,
                 calculate_drone_to_drone: bool, calculate_drone_to_target: bool, render_mode: str,
                 n_phi: int = 36, n_theta: int = 36, ray_max_distance: float = 50,
                 enable_ray: bool = False, **kwargs):
        """
        Initializes the multi-agent environment with the given configuration.

        Args:
            num_agents (int): The number of drone agents in the environment.
            DroneClass (Type[Drone]): The drone class to be used for creating drone instances.
            num_targets (int): The number of targets in the environment.
            world_bounds (np.ndarray): The boundaries within which drones can operate.
            respawn_box (np.ndarray): The box within which drones can be respawned.
            spawn_angles (np.ndarray): The range of angles for drones' initial orientations.
            camera_type (str): The type of camera used by drones ('stereo', 'mono', or None).
            depth (bool): Flag indicating whether depth perception is enabled.
            calculate_drone_to_drone (bool): Flag indicating whether to calculate drone-to-drone relational data.
            calculate_drone_to_target (bool): Flag indicating whether to calculate drone-to-target relational data.
            **kwargs: Additional keyword arguments for drone initialization.
        """
        assert camera_type in ["stereo", "mono", None], "Invalid camera type, must be one of 'stereo', 'mono' or None"
        # region Model Initialization
        mj_model = get_model(
            num_agents=num_agents,
            num_targets=num_targets,
            map_bounds=world_bounds,
            light_height=kwargs.get("light_height", 5),
            spacing=kwargs.get("spacing", 2),
            drone_height=kwargs.get("drone_height", 0.5),
            target_dimensions_range=kwargs.get("target_dimensions_range",
                                               np.array([[0.05, 0.05, 0.05], [0.2, 0.2, 0.2]])),
        )
        mj_model.opt.timestep = kwargs.get("timestep", 0.01)
        
        self.model: MjModel = mj_model
        self.data: MjData = MjData(self.model)
        self.render_mode = render_mode
        assert render_mode in ["lidar", "mujoco", None], \
            "Invalid render mode, must be one of 'lidar', 'drone_pov', 'free_pov', or None"
        self.renderer: Renderer = Renderer(self.model, kwargs.get("height", 480), kwargs.get("width", 640))
        self.handler: viewer.Handle = (
            viewer.launch_passive(self.model, self.data)) if render_mode in ["lidar", "drone_pov"] else None
        self.depth = depth
        self.calculate_drone_to_drone = calculate_drone_to_drone
        self.calculate_drone_to_target = calculate_drone_to_target
        self.world_bounds = world_bounds if world_bounds is not None else np.array([[-25, -25, 0], [25, 25, 10]])
        self.max_distance = np.linalg.norm(self.world_bounds[1] - self.world_bounds[0])
        self.noise = kwargs.get("noise", 0.01)
        
        self.camera_type = camera_type
        self.num_agents = num_agents
        self.agent_ids: Set[AgentID] = set(range(num_agents))
        self.target_ids = set(range(num_targets))
        # endregion
        
        # region Drone and Target Initialization
        
        self.drone_to_drone_distance = np.zeros((num_agents, num_agents)) if calculate_drone_to_drone else None
        self.drone_to_drone_sin_theta = np.zeros((num_agents, num_agents)) if calculate_drone_to_drone else None
        self.drone_to_drone_cos_theta = np.zeros((num_agents, num_agents)) if calculate_drone_to_drone else None
        self.drone_to_drone_cartesian = np.zeros((num_agents, num_agents, 3)) if calculate_drone_to_drone else None
        
        self.drone_to_target_distance = np.zeros((num_agents, num_targets)) if calculate_drone_to_target else None
        self.drone_to_target_sin_theta = np.zeros((num_agents, num_targets)) if calculate_drone_to_target else None
        self.drone_to_target_cos_theta = np.zeros((num_agents, num_targets)) if calculate_drone_to_target else None
        self.drone_to_target_cartesian = np.zeros((num_agents, num_targets, 3)) if calculate_drone_to_target else None
        
        self.drones = {agent_id: DroneClass(
            model=self.model,
            data=self.data,
            agent_id=agent_id,
            renderer=self.renderer,
            camera_type=self.camera_type,
            spawn_box=respawn_box,
            spawn_angles=spawn_angles,
            bullet_velocity=kwargs.get("bullet_velocity", 10),
            drone_to_drone_distance=self.drone_to_drone_distance[agent_id] if calculate_drone_to_drone else None,
            drone_to_target_distance=self.drone_to_target_distance[agent_id] if calculate_drone_to_target else None,
            drone_to_drone_sin_theta=self.drone_to_drone_sin_theta[agent_id] if calculate_drone_to_drone else None,
            drone_to_target_sin_theta=self.drone_to_target_sin_theta[agent_id] if calculate_drone_to_target else None,
            drone_to_drone_cos_theta=self.drone_to_drone_cos_theta[agent_id] if calculate_drone_to_drone else None,
            drone_to_target_cos_theta=self.drone_to_target_cos_theta[agent_id] if calculate_drone_to_target else None,
            drone_to_drone_cartesian=self.drone_to_drone_cartesian[agent_id] if calculate_drone_to_drone else None,
            drone_to_target_cartesian=self.drone_to_target_cartesian[agent_id] if calculate_drone_to_target else None,
            n_phi=n_phi,
            n_theta=n_theta,
            ray_max_distance=ray_max_distance,
            enable_ray=enable_ray,
            max_world_diagonal=np.linalg.norm(self.world_bounds[1] - self.world_bounds[0]),
            **kwargs
        ) for agent_id in self.agent_ids}
        self.targets = {target_id: Target(
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
        
        # region Space Initialization
        self.observation_space: SpaceDict[AgentID, Space[ObsType]] = SpaceDict({
            agent_id: drone.observation_space for agent_id, drone in self.drones.items()
        })
        self.action_space: SpaceDict[AgentID, Space[ObsType]] = SpaceDict({
            agent_id: drone.action_space for agent_id, drone in self.drones.items()
        })
        # endregion
        
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
        self.floor_geom: _MjModelGeomViews = self.model.geom("floor")
        self.floor_geom_id = self.floor_geom.id
        # endregion
        
        # region Pre-computation for rendering
        self.frame_skip = kwargs.get("frame_skip", 1)
        self.fps = kwargs.get("fps", 60)
        self.render_every = max(1, int((self.fps * self.frame_skip) / self.fps))
        self.physics_step_index = 0
        # endregion
    
    def collisions(self) -> None:
        """
        Checks and processes collisions between drones, bullets, targets, and the environment.
        """
        
        # region Initialize Collision Sets
        drones_shooting_drones = set()
        drones_hit_by_bullet = set()
        drones_hit_floor = set()
        drones_crash_target = set()
        drones_hit_target = set()
        targets_hit = set()
        bullet_floor_contacts = set()
        # endregion
        
        # region Collision Checks
        for contact in self.data.contact:
            geom_1_id, geom_2_id = contact.geom1, contact.geom2
            
            # region Bullet - Drone collision checks
            bullet_1 = self._bullet_geom_ids_to_drones.get(geom_1_id)
            drone_1 = self._drone_geom_ids_to_drones.get(geom_2_id)
            target_1 = self._target_geom_ids_to_targets.get(geom_2_id)
            floor_1 = geom_2_id == self.floor_geom_id
            
            if bullet_1:
                if drone_1 and bullet_1.agent_id != drone_1.agent_id:
                    drones_shooting_drones.add(bullet_1.agent_id)
                    drones_hit_by_bullet.add(drone_1.agent_id)
                elif target_1:
                    drones_crash_target.add(bullet_1.agent_id)
                    target_1.reset()
                elif floor_1:
                    bullet_floor_contacts.add(bullet_1.agent_id)
            
            bullet_2 = self._bullet_geom_ids_to_drones.get(geom_2_id)
            drone_2 = self._drone_geom_ids_to_drones.get(geom_1_id)
            target_2 = self._target_geom_ids_to_targets.get(geom_1_id)
            floor_2 = geom_1_id == self.floor_geom_id
            
            if bullet_2:
                if drone_2 and bullet_2.agent_id != drone_2.agent_id:
                    drones_shooting_drones.add(bullet_2.agent_id)
                    drones_hit_by_bullet.add(drone_2.agent_id)
                elif target_2:
                    drones_crash_target.add(bullet_2.agent_id)
                    target_2.reset()
                elif floor_2:
                    bullet_floor_contacts.add(bullet_2.agent_id)
            # endregion
            
            # region Drone - Target collision checks
            if drone_1 and target_1:
                drones_hit_target.add(drone_1.agent_id)
                targets_hit.add(target_1.target_id)
            if drone_2 and target_2:
                drones_hit_target.add(drone_2.agent_id)
                targets_hit.add(target_2.target_id)
            # endregion
            
            # region Drone - Floor collision checks
            if drone_1 and floor_1:
                drones_hit_floor.add(drone_1.agent_id)
            if drone_2 and floor_2:
                drones_hit_floor.add(drone_2.agent_id)
            # endregion
        # endregion
        
        # region Set Drone Flags and Reset Targets and Bullets
        for agent_id in drones_shooting_drones:
            self.drones[agent_id].shot_drone = True
        for agent_id in drones_hit_by_bullet:
            self.drones[agent_id].got_shot = True
        for agent_id in drones_hit_floor:
            self.drones[agent_id].hit_floor = True
        for agent_id in drones_crash_target:
            self.drones[agent_id].crash_target = True
        for agent_id in drones_hit_target:
            self.drones[agent_id].shot_target = True
        for target_id in targets_hit:
            self.targets[target_id].reset()
        for agent_id in bullet_floor_contacts:
            self.drones[agent_id].bullet.reset()
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
        Updates the relational data between drones and between drones and targets. This function computes
        the normalized distances, angles (and their sine and cosine values), and Cartesian coordinates
        for each drone-drone and drone-target pair. The computed values are normalized with respect to
        the maximum distance in the environment and stored in corresponding class attributes.

        The function handles two main types of relationships:
        1. Drone-to-Drone: Computes the relative positions and orientations between each pair of drones.
        2. Drone-to-Target: Computes the relative positions and orientations between each drone and target.

        The calculations are as follows:

        For Drone-to-Drone:
        - Distance: \(d_{ij} = \frac{\| \mathbf{p}_i - \mathbf{p}_j \|}{\text{max\_distance}}\)
        - Angle (Theta): \(\theta_{ij} = \text{atan2}(y_{ij}, x_{ij})\)
        - Sine of Theta: \(\sin(\theta_{ij})\)
        - Cosine of Theta: \(\cos(\theta_{ij})\)
        - Cartesian Coordinates: \(\Delta \mathbf{p}_{ij} = \frac{\mathbf{p}_i - \mathbf{p}_j}{\text{max\_distance}} + \mathcal{N}(0, \text{noise})\)

        For Drone-to-Target:
        - Distance: \(d_{ik} = \frac{\| \mathbf{p}_i - \mathbf{t}_k \|}{\text{max\_distance}}\)
        - Angle (Theta): \(\theta_{ik} = \text{atan2}(y_{ik}, x_{ik})\)
        - Sine of Theta: \(\sin(\theta_{ik})\)
        - Cosine of Theta: \(\cos(\theta_{ik})\)
        - Cartesian Coordinates: \(\Delta \mathbf{p}_{ik} = \frac{\mathbf{p}_i - \mathbf{t}_k}{\text{max\_distance}} + \mathcal{N}(0, \text{noise})\)

        where:
        - \( \mathbf{p}_i \) and \( \mathbf{p}_j \) are the positions of drones i and j, respectively.
        - \( \mathbf{t}_k \) is the position of target k.
        - \(\mathcal{N}(0, \text{noise})\) represents Gaussian noise with a mean of 0 and a standard deviation defined by the `noise` attribute.
        - \(\text{max\_distance}\) is the maximum possible distance in the environment, used for normalization.

        Note:
        - The distances and Cartesian coordinates are normalized by the maximum possible distance in the environment to ensure they are within a [0, 1] range.
        - The function updates class attributes that store the computed relational data, which can then be used for downstream tasks like observation generation or reward calculation.
        """
        drone_positions = np.array([self.data.xpos[self.drones[agent_id].body_id] for agent_id in self.agent_ids])
        if self.calculate_drone_to_drone:
            deltas = drone_positions[:, np.newaxis, :] - drone_positions[np.newaxis, :, :]
            self.drone_to_drone_distance[:, :] = np.linalg.norm(deltas, axis=2) / self.max_distance
            np.fill_diagonal(self.drone_to_drone_distance, 0)  # Avoid self-distance
            thetas = np.arctan2(deltas[..., 1], deltas[..., 0])
            self.drone_to_drone_sin_theta[:, :] = np.sin(thetas)
            self.drone_to_drone_cos_theta[:, :] = np.cos(thetas)
            self.drone_to_drone_cartesian[:, :, :] = (deltas / self.max_distance +
                                                      np.random.normal(0, self.noise, deltas.shape))
        
        if self.calculate_drone_to_target:
            target_positions = np.array(
                [self.data.geom_xpos[self.targets[target_id].geom_id] for target_id in self.target_ids])
            delta_targets = drone_positions[:, np.newaxis, :] - target_positions[np.newaxis, :, :]
            self.drone_to_target_distance[:, :] = np.linalg.norm(delta_targets, axis=2) / self.max_distance
            thetas_targets = np.arctan2(delta_targets[..., 1], delta_targets[..., 0])
            self.drone_to_target_sin_theta[:, :] = np.sin(thetas_targets)
            self.drone_to_target_cos_theta[:, :] = np.cos(thetas_targets)
            self.drone_to_target_cartesian[:, :, :] = (delta_targets / self.max_distance +
                                                       np.random.normal(0, self.noise, delta_targets.shape))
    
    def update_out_of_bounds(self) -> None:
        """
        Checks and updates the out-of-bounds status for each drone.
        """
        drone_positions = np.array([self.data.body_xpos[drone.body_id][:3] for drone in self.drones.values()])
        out_of_bounds_x = np.logical_or(drone_positions[:, 0] < self.world_bounds[0, 0],
                                        drone_positions[:, 0] > self.world_bounds[1, 0])
        out_of_bounds_y = np.logical_or(drone_positions[:, 1] < self.world_bounds[0, 1],
                                        drone_positions[:, 1] > self.world_bounds[1, 1])
        out_of_bounds_z = np.logical_or(drone_positions[:, 2] < self.world_bounds[0, 2],
                                        drone_positions[:, 2] > self.world_bounds[1, 2])
        out_of_bounds = np.logical_or(np.logical_or(out_of_bounds_x, out_of_bounds_y), out_of_bounds_z)
        for idx, drone in enumerate(self.drones.values()):
            drone.out_of_bounds = out_of_bounds[idx]
    
    def update_drones(self) -> None:
        """
        Updates the state of the drones based on the current environment.
        """
        if self.physics_step_index % self.render_every == 0:
            for drone in self.drones.values():
                drone.update_rays()
    
    @property
    def step_results(self) -> Tuple[MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict]:
        """
        Compiles the results of the step function.
        """
        observation = self.observation(render=self.physics_step_index % self.render_every == 0)
        reward = self.reward
        truncated = self.truncated
        done = self.done
        info = self.log_info
        self.physics_step_index += 1
        return observation, reward, truncated, done, info
    
    def step(self, action_dict: ActType) -> Tuple[MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict]:
        """
        Advances the environment by one time step based on the actions of each agent.
        """
        self.reset_agent_flags()
        self.act(action_dict)
        mj_step(self.model, self.data, self.frame_skip)
        self.update_drone_relations()
        self.collisions()
        self.update_drones()
        return self.step_results
    
    def render(self, mode: str) -> None:
        """
        Renders the current state of the environment.
        """
        assert mode in ["lidar", "mujoco", None], \
            "Invalid rendering mode, must be one of 'lidar', 'drone_follow', 'free_cam' or None"
        
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
        return self.observation(render=True), self.reward
    
    def close(self):
        """
        Closes the environment and releases resources.
        """
        if self.handler is not None:
            self.handler.close()
    
    def observation(self, render: bool = False) -> MultiAgentDict:
        """
        Retrieves observations for each drone in the environment.
        """
        return {agent_id: self.drones[agent_id].observe(render) for agent_id in self.agent_ids}
    
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
