from __future__ import annotations

from abc import abstractmethod
from typing import Union, Tuple, Optional

import numpy as np
import quaternion
from gymnasium import Space
from ray.rllib.utils.typing import EnvActionType, EnvObsType, AgentID
from scipy.spatial.transform import Rotation as R
from mujoco import mj_multiRay, mjMAXVAL, mj_name2id, mjtObj, Renderer, MjData, MjModel


class BaseBullet:
    """
    Represents a bullet object in the simulation environment, managing its properties and interactions.

    Attributes:
        agent_id (AgentID): Identifier of the agent controlling this bullet.
        model (MjModel): The MuJoCo model associated with the simulation.
        data (MjData): The data structure containing the current state of the simulation.
        bullet_id (int): Unique identifier for the bullet within the simulation.
        bullet_velocity (float): The velocity at which the bullet is shot.
        parent (BaseDrone): The drone object that fired this bullet.
        body_id (int): The ID of the bullet's body within the MuJoCo model.
        geom_id (int): The ID of the bullet's geometry within the MuJoCo model.
        spawn_site_id (int): The ID of the site where the bullet is spawned.
        joint_id (int): The ID of the joint associated with the bullet.
        bullet_color (np.ndarray): The RGBA color of the bullet.
    """
    
    def __init__(self, agent_id: AgentID, model: MjModel, data: MjData, bullet_id: int, bullet_velocity: float,
                 parent: BaseDrone, bullet_color: np.ndarray = None):
        """
        Initializes a new instance of the BaseBullet class.

        Arguments:
            :param agent_id: Identifier of the agent controlling this bullet.
            :param model: The MuJoCo model associated with the simulation.
            :param data: The data structure containing the current state of the simulation.
            :param bullet_id: Unique identifier for the bullet within the simulation.
            :param bullet_velocity: The velocity at which the bullet is shot.
            :param parent: The drone object that fired this bullet.
            :param bullet_color: The RGBA color of the bullet.
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
        self.spawn_site_id = mj_name2id(model, mjtObj.mjOBJ_SITE, f"bullet_spawn_position_{bullet_id}")
        self.joint_id = mj_name2id(model, mjtObj.mjOBJ_JOINT, f"bullet_{bullet_id}")
        self.jnt_qp_adr = self.model.jnt_qposadr[self.joint_id]
        self.jnt_qv_adr = self.model.jnt_dofadr[self.joint_id]
        self.bullet_color = bullet_color if bullet_color is not None else [.2, 0, .2, 1]
        self.model.geom_rgba[self.geom_id] = self.bullet_color
    
    def shoot(self):
        """
        Simulates the shooting action of the bullet.
        """
        quat = self.parent.data.xquat[self.parent.body_id]
        
        # Check if the quaternion is valid (non-degenerate and normalized)
        is_non_degenerate = not np.all(quat == 0)
        is_normalized = np.isclose(np.linalg.norm(quat), 1.0)
        
        if not self.model.body_contype[self.body_id] and is_non_degenerate and is_normalized:
            # Set the bullet's position to the spawn site's position
            self.data.qpos[self.jnt_qp_adr: self.jnt_qp_adr + 3] = self.data.site_xpos[self.spawn_site_id]
            
            # Calculate the bullet's velocity
            direction = R.from_quat(quat).apply([1, 0, 0])
            velocity = direction * self.bullet_velocity
            
            # Set the bullet's velocity
            self.data.qvel[self.jnt_qv_adr: self.jnt_qv_adr + 3] = velocity
            
            # Update the bullet's color to indicate it's active
            self.model.geom_rgba[self.geom_id] = self.bullet_color
            
            # Activate the bullet by setting its contact type and affinity
            self.model.body_contype[self.body_id] = 1
            self.model.body_conaffinity[self.body_id] = 1
    
    def reset(self):
        """
        Resets the bullet's state within the simulation.
        """
        self.data.qpos[self.jnt_qp_adr: self.jnt_qp_adr + 7].fill(0)
        self.data.qpos[self.jnt_qp_adr + 6].fill(1)
        self.data.qvel[self.jnt_qv_adr: self.jnt_qv_adr + 6].fill(0)
        self.model.geom_rgba[self.geom_id] = [0, 0, 0, 0]
        self.model.body_contype[self.body_id] = 0
        self.model.body_conaffinity[self.body_id] = 0


class BaseTarget:
    """
    Represents a target object in the simulation environment, managing its properties and behavior.

    Attributes:
        model (MjModel): The MuJoCo model associated with the simulation.
        data (MjData): The data structure containing the current state of the simulation.
        target_id (int): Unique identifier for the target within the simulation.
        spawn_box (np.ndarray): The bounding box within which the target can be spawned.
        spawn_angles (np.ndarray): The range of angles for random orientation of the target.
        geom_id (int): The ID of the target's geometry within the MuJoCo model.
    """
    
    def __init__(self, model: MjModel, data: MjData, target_id: int, spawn_box: np.ndarray,
                 spawn_angles: np.ndarray, target_color: np.ndarray = np.array([0, 1, 0, 1])):
        """
        Initializes a new instance of the BaseTarget class.

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


class BaseDrone:
    """
    The BaseDrone class serves as a foundation for creating diverse drone simulations within the MuJoCo environment.
    It provides a versatile framework that users can extend to develop customized drone behaviors, integrating various
    sensory inputs and dynamic interactions within the simulation space.

    **Extendibility:**

    Users can inherit from this class to create specialized drone models. By overriding the abstract methods and
    utilizing the provided initialization parameters, users can define unique observation spaces, action spaces,
    reward mechanisms, and more. The class is designed to facilitate the addition of custom sensors, actuators,
    and behavioral logic, enabling a wide range of drone simulations.

    **Sensory Tools:**

    - **Rays:** Utilize ray casting to simulate lidar-like sensors, enabling obstacle detection and distance measurement.
    - **Camera:** Incorporate mono or stereo cameras to capture visual inputs from the drone's perspective. Depth perception can also be enabled for depth-aware applications.
    - **Relative Positioning:** Access precomputed relative positions and angles between drones and between drones and targets, facilitating spatial awareness and interaction within the environment.

    **Customization Options:**

    - **Custom Flags:** Define and manage custom flags to track drone-specific conditions or states throughout the simulation.
    - **Custom Update Function:** Implement a custom update logic within the :meth:`custom_update` method to define the drone's behavior at each simulation step, including sensor updates, state transitions, and more.
    - **Reset Custom Flags:** Provide a mechanism to reset custom flags, ensuring proper initialization at the beginning of each simulation step or episode.

    :param model: The MuJoCo model associated with the drone.
    :type model: MjModel
    :param data: The data structure containing the current state of the simulation.
    :type data: MjData
    :param agent_id: Unique identifier for the drone within the simulation.
    :type agent_id: AgentID
    :param spawn_box: Defines the volume in which the drone can be respawned.
    :type spawn_box: np.ndarray
    :param spawn_angles: Specifies the range of angles for the drone's initial orientation.
    :type spawn_angles: np.ndarray
    :param bullet_velocity: The velocity at which the drone's bullets are shot.
    :type bullet_velocity: float
    :param max_world_diagonal: Maximum diagonal distance in the world, used for normalization.
    :type max_world_diagonal: float
    :param bullet_color: The RGBA color of the drone's bullet.
    :type bullet_color: np.ndarray, optional

    **Example Usage:**

    To create a custom drone class, inherit from `BaseDrone` and implement the required abstract methods. Initialize
    the parent class with the desired configuration, and extend the functionality by adding custom sensors, action
    logic, or state management routines.

    **Note:**

    This class is abstract and is not intended to be instantiated directly. Users should create subclasses that
    implement the required abstract methods and may add additional methods and properties as needed.
    """

    def __init__(
            self,
            model: MjModel,
            data: MjData,
            agent_id: AgentID,
            spawn_box: np.ndarray,
            spawn_angles: np.ndarray,
            bullet_velocity: float,
            drone_to_drone_distance: Optional[np.ndarray] = None,
            drone_to_target_distance: Optional[np.ndarray] = None,
            drone_to_drone_sin_theta: Optional[np.ndarray] = None,
            drone_to_target_sin_theta: Optional[np.ndarray] = None,
            drone_to_drone_cos_theta: Optional[np.ndarray] = None,
            drone_to_target_cos_theta: Optional[np.ndarray] = None,
            drone_to_drone_theta: Optional[np.ndarray] = None,
            drone_to_target_theta: Optional[np.ndarray] = None,
            drone_to_drone_cartesian: Optional[np.ndarray] = None,
            drone_to_target_cartesian: Optional[np.ndarray] = None,
            max_world_diagonal: float = 100,
            bullet_color: np.ndarray = None
    ):
        """
        Initializes a new BaseDrone instance, setting up its environment, sensors, actuators, and other properties.

        Parameters:
            model (MjModel): The MuJoCo model object associated with the simulation.
            data (MjData): The MuJoCo data object containing the current simulation state.
            agent_id (AgentID): A unique identifier for the drone within the simulation.
            spawn_box (np.ndarray): A 2D array defining the volume in which the drone can be respawned
                ([min_x, min_y, min_z], [max_x, max_y, max_z]).
            spawn_angles (np.ndarray): A 2D array specifying the range of angles for the drone's initial orientation
                ([min_roll, min_pitch, min_yaw], [max_roll, max_pitch, max_yaw]).
            bullet_velocity (float): The velocity at which the drone's bullets are fired.
            max_world_diagonal (float): The maximum diagonal distance across the simulation world, used for
                normalization.

            Optional Parameters:
            drone_to_drone_distance (Optional[np.ndarray]): Precomputed distances between this drone and other drones,
                if any.
            drone_to_target_distance (Optional[np.ndarray]): Precomputed distances between this drone and targets, if
                any.
            drone_to_drone_sin_theta (Optional[np.ndarray]): Precomputed sine values of angles between this drone and
                others.
            drone_to_target_sin_theta (Optional[np.ndarray]): Precomputed sine values of angles between this drone and
                targets.
            drone_to_drone_cos_theta (Optional[np.ndarray]): Precomputed cosine values of angles between this drone and
                others.
            drone_to_target_cos_theta (Optional[np.ndarray]): Precomputed cosine values of angles between this drone
                and targets.
            drone_to_drone_cartesian (Optional[np.ndarray]): Precomputed Cartesian coordinates of other drones relative
                to this one.
            drone_to_target_cartesian (Optional[np.ndarray]): Precomputed Cartesian coordinates of targets relative to
                this drone.
            bullet_color (Optional[np.ndarray]): RGBA color for the drone's bullets. Defaults to red if not provided.

        Note:
            This is an abstract drones class and is meant to be subclassed. It does not implement any specific drone behavior.
        """
        
        # region Basic Initialization
        self.renderer: Renderer = None
        self.model = model
        self.data = data
        self.agent_id = agent_id
        self.camera_type = None
        self.depth = False
        self.spawn_box = spawn_box
        self.spawn_angles = spawn_angles
        self.bullet_velocity = bullet_velocity
        self.max_world_diagonal = max_world_diagonal
        # endregion
        
        # region BaseDrone-to-BaseDrone and BaseDrone-to-BaseTarget Distances and Angles
        self.drone_to_drone_distance = drone_to_drone_distance
        self.drone_to_target_distance = drone_to_target_distance
        self.drone_to_drone_sin_theta = drone_to_drone_sin_theta
        self.drone_to_target_sin_theta = drone_to_target_sin_theta
        self.drone_to_drone_cos_theta = drone_to_drone_cos_theta
        self.drone_to_target_cos_theta = drone_to_target_cos_theta
        self.drone_to_drone_theta = drone_to_drone_theta
        self.drone_to_target_theta = drone_to_target_theta
        self.drone_to_drone_cartesian = drone_to_drone_cartesian
        self.drone_to_target_cartesian = drone_to_target_cartesian
        # endregion
        
        # region Ray Unit Vectors and Related Attributes
        self.enable_ray = False
        self.initial_ray_unit_vectors: np.ndarray = None
        self.flattened_ray_unit_vectors: np.ndarray = None
        self.distances: np.ndarray = None
        self.intersecting_geoms: np.ndarray = None
        self.ray_max_distance: np.ndarray = None
        self.num_rays: int = None
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
        self.mono_camera_id: int = None
        self.left_camera_id: int = None
        self.right_camera_id: int = None
        # endregion
        
        # region Sensor IDs
        self.accel_id = mj_name2id(model, mjtObj.mjOBJ_SENSOR, f"accelerometer_{agent_id}")
        self.gyro_id = mj_name2id(model, mjtObj.mjOBJ_SENSOR, f"gyro_{agent_id}")
        # endregion
        
        # region Initialize Other Attributes
        self.initial_pos = self.data.xpos[self.body_id].copy()
        self.bullet = BaseBullet(agent_id, model, data, agent_id, bullet_velocity, self,
                                 bullet_color=bullet_color or [1, 0, 0, 1])
        self.front_left_actuator_id = mj_name2id(model, mjtObj.mjOBJ_ACTUATOR, f"front_left_{self.agent_id}")
        self.front_right_actuator_id = mj_name2id(model, mjtObj.mjOBJ_ACTUATOR, f"front_right_{self.agent_id}")
        self.back_left_actuator_id = mj_name2id(model, mjtObj.mjOBJ_ACTUATOR, f"back_left_{self.agent_id}")
        self.back_right_actuator_id = mj_name2id(model, mjtObj.mjOBJ_ACTUATOR, f"back_right_{self.agent_id}")
        self.actuator_ids = np.array([
            self.front_left_actuator_id,
            self.front_right_actuator_id,
            self.back_left_actuator_id,
            self.back_right_actuator_id
        ])
        # endregion
        
        # region Image Arrays
        self._image_1: np.ndarray = None
        self._image_2: np.ndarray = None
        # endregion
        
        # region Flags for Various Conditions
        self.hit_floor = False
        self.got_shot = False
        self.shot_target = False
        self.shot_drone = False
        self.out_of_bounds = False
        self.crash_target = False
        self.crash_drone = False
        self.bullet_out_of_bounds = False
        self.just_shot = False
        # endregion
    
    # region Ray Casting and Camera Methods
    def init_rays(self, n_phi: int, n_theta: int, ray_max_distance: float):
        """
        Initializes ray casting properties for the drone, used for simulating lidar-like sensors.

        Parameters:
            n_phi (int): The number of discretization steps in the azimuthal (horizontal) direction.
            n_theta (int): The number of discretization steps in the polar (vertical) direction.
            ray_max_distance (float): The maximum distance each ray can detect.

        Note:
            This method should be called if ray-based sensing is required for the drone. It initializes the unit vectors
            for the rays based on the specified discretization and the maximum detectable distance for each ray.
        """
        self.enable_ray = True
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
        self.update_rays()
    
    def init_camera(self, camera_type: str, renderer: Renderer, depth: bool = False):
        """
        Initializes the camera for the drone, used for capturing images from the drone's perspective.

        Parameters:
            camera_type (str): The type of camera, either 'mono' for a single camera or 'stereo' for a pair of cameras simulating stereo vision.
            renderer (Renderer): The renderer instance associated with the simulation for rendering the camera images.
            depth (bool): Indicates whether depth information should be included in the captured images.

        Raises:
            ValueError: If an invalid camera_type is provided.

        Note:
            The camera(s) initialized by this method can be used to simulate visual input for the drone. The method sets up
            the necessary camera ID(s) and initializes the image arrays based on the specified camera configuration and whether
            depth sensing is enabled.
        """
        assert camera_type in ["mono", "stereo"], "Camera type must be either 'mono' or 'stereo'."
        self.camera_type = camera_type
        self.depth = depth
        self.renderer = renderer
        drone_name = f"drone_{self.agent_id}"
        if self.camera_type == "mono":
            self.mono_camera_id = mj_name2id(self.model, mjtObj.mjOBJ_CAMERA, drone_name)
        elif self.camera_type == "stereo":
            self.left_camera_id = mj_name2id(self.model, mjtObj.mjOBJ_CAMERA, f"{drone_name}_left")
            self.right_camera_id = mj_name2id(self.model, mjtObj.mjOBJ_CAMERA, f"{drone_name}_right")
            
        dimensions = (self.renderer.height, self.renderer.width, 3) if not self.depth else (
            self.renderer.height, self.renderer.width)
        self._image_1 = np.zeros(dimensions)
        if self.camera_type == "stereo":
            self._image_2 = np.zeros(dimensions)
    
    @property
    def ray_distances(self) -> np.ndarray:
        """
        Computes and returns the normalized distances detected by the drone's rays.

        Returns:
            np.ndarray: An array of normalized distances for each ray.

        Note:
            This property should be accessed after the rays have been updated in the simulation step to get the latest
            distance measurements.
        """
        return self.distances / self.ray_max_distance
    
    def update_images(self) -> None:
        """
        Captures and updates the images from the drone's camera(s).

        Raises:
            ValueError: If the camera has not been initialized before calling this method.

        Note:
            This method should be called at each simulation step where visual input is needed. It updates the drone's
            image arrays with the latest views captured by the camera(s).
        """
        if not self.camera_type:
            raise ValueError("Camera type not initialized. Call init_camera before capturing images.")
        
        if self.camera_type == "mono":
            self.renderer.update_scene(self.data, camera=self.mono_camera_id)
            self.renderer.render(out=self._image_1)
        elif self.camera_type == "stereo":
            self.renderer.update_scene(self.data, camera=self.left_camera_id)
            self.renderer.render(out=self._image_1)
            self.renderer.update_scene(self.data, camera=self.right_camera_id)
            self.renderer.render(out=self._image_2)
    
    @property
    def get_lidar_data(self) -> np.ndarray:
        """
        Computes and returns the lidar-like data based on the ray distances and their unit vectors.

        Returns:
            np.ndarray: An array representing the lidar-like data, combining the ray unit vectors and their measured
            distances.

        Note:
            This method provides a way to simulate lidar sensor data, which can be used for obstacle detection,
            navigation, etc.
        """
        return self.distances * self.flattened_ray_unit_vectors.reshape(self.num_rays, 3)
    
    def update_rays(self):
        """
        Updates the rays based on the drone's current orientation and position in the simulation.

        Note:
            This method should be called at each simulation step to update the ray casting based on the drone's
            current state, allowing for dynamic detection of obstacles and other objects in the environment.
        """
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
        np.divide(self.distances, self.ray_max_distance, out=self.distances)    # Normalize distances
    # endregion
    
    # region Update and Reset Methods
    def update(self, render: bool = False):
        """
        Updates the drone's state, including sensor readings and camera images.

        Parameters:
            render (bool): Specifies whether to update sensor readings that require rendering (e.g., cameras, lidar).

        Note:
            This method encapsulates the updates for various sensors and should be called every simulation step.
            It allows for the integration of sensor updates within the drone's control loop.
        """
        if self.enable_ray and render:
            self.update_rays()
        if self.camera_type and render:
            self.update_images()
        self.custom_update()
    
    def respawn(self):
        """
        Respawns the drone at a new location within the specified spawn box and with a random orientation from the
        specified angles.

        Note:
            This method is typically called when resetting the simulation or when the drone needs to be repositioned to
            a new start location.
        """
        pos = np.random.uniform(*self.spawn_box) + self.initial_pos[:3]
        quat = quaternion.from_euler_angles(np.random.uniform(*self.spawn_angles))
        self.data.qpos[self.free_joint_qpos_address: self.free_joint_qpos_address + 7] = np.concatenate(
            [pos, quaternion.as_float_array(quat)])
        self.data.qvel[self.free_joint_qvel_address: self.free_joint_qvel_address + 6].fill(0)
        self.bullet.reset()
    
    def reset(self):
        """
        Resets the drone's state to initial conditions, including position, orientation, and sensor readings.

        Note:
            This method should be called to reinitialize the drone's state at the beginning of each new episode or
            simulation run.
        """
        self.bullet.reset()
        self.reset_custom_flags()
        self.reset_default_flags()
        self.respawn()
    
    def reset_default_flags(self):
        """
        Resets the default flags indicating various drone states (e.g., collision, out of bounds).

        Note:
            This is a helper method to clear all the flags at the beginning of each simulation step or episode.
        """
        self.hit_floor = False
        self.got_shot = False
        self.shot_target = False
        self.shot_drone = False
        self.out_of_bounds = False
        self.crash_target = False
        self.crash_drone = False
        self.bullet_out_of_bounds = False
        self.just_shot = False
    # endregion
    
    # region Abstract Methods
    # region Observation and Action Spaces
    @abstractmethod
    def observation_space(self) -> Space[EnvObsType]:
        """
        Abstract method that should be implemented in subclasses to define the drone's observation space.

        Returns:
            Space[EnvObsType]: The observation space of the drone.
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def observation(self) -> EnvObsType:
        """
        Retrieves the drone's observations at the current simulation step.
        
        Returns:
            EnvObsType: The drone's observations.
        """
        raise NotImplementedError
    
    @abstractmethod
    def action_space(self) -> Space[EnvActionType]:
        """
        Abstract method that should be implemented in subclasses to define the drone's action space.

        Returns:
            Space[EnvActionType]: The action space of the drone, specifying all possible actions it can take.
        """
        raise NotImplementedError
    
    @abstractmethod
    def act(self, action: EnvActionType):
        """
        Executes the given action in the simulation. This method should be implemented in subclasses to specify
        how the drone responds to actions.

        Parameters:
            action (EnvActionType): The action to execute, which should be compatible with the drone's action space.
        """
        raise NotImplementedError
    # endregion
    
    # region Reward, Log Info, and Episode Termination
    @property
    @abstractmethod
    def reward(self) -> float:
        """
        Calculates and returns the reward for the current state of the drone. This method should be implemented
        in subclasses based on the specific task or objective.

        Returns:
            float: The calculated reward for the drone's current state.
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def log_info(self) -> dict:
        """
        Returns a dictionary of additional information about the drone's current state. This method can be used
        to provide extra details for logging or debugging purposes.

        Returns:
            dict: A dictionary containing additional information about the drone's state.
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def done(self) -> bool:
        """
        Determines whether the episode has ended for the drone. This method should be implemented in subclasses
        to specify the termination conditions.

        Returns:
            bool: True if the episode is complete, otherwise False.
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def truncated(self) -> bool:
        """
        Checks if the episode is truncated for the drone, which can occur if the episode ends before the natural
        termination condition is met. Implement this method to specify truncation conditions.

        Returns:
            bool: True if the episode is truncated, otherwise False.
        """
        raise NotImplementedError
    # endregion
    
    @abstractmethod
    def reset_custom_flags(self):
        """
        Resets any custom flags or indicators specific to the drone implementation. This method should be implemented
        in subclasses to clear any custom status indicators at the start of a new episode or simulation step.
        """
        raise NotImplementedError
    
    @abstractmethod
    def custom_update(self):
        """
        Contains custom update logic for the drone, which may include updating internal states, processing sensor data,
        or other tasks. Implement this method in subclasses to specify any additional update behaviors needed for the drone.
        """
        raise NotImplementedError
    # endregion
