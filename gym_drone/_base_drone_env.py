# region Imports and global variables
import os
from abc import ABC, abstractmethod
from typing import SupportsFloat, Any, Tuple

import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Space, Box, Dict
from gymnasium.utils import EzPickle
from mujoco import MjModel, MjData, mj_step
from mujoco._structs import _MjDataBodyViews
import quaternion
from numpy.random import default_rng

XML_PATH = os.path.join(os.path.dirname(__file__), "assets/UAV/scene.xml")


# endregion


class Drone:
    def __init__(self,
                 data: MjData,
                 spawn_box: np.ndarray,
                 spawn_max_velocity: float,
                 rng: np.random.Generator = default_rng()
                 ):
        """
        Initialize the Drone object
        :param data:
        :param spawn_box:
        :param spawn_max_velocity:
        """
        self.data = data
        self.body: _MjDataBodyViews = data.body('drone')
        self.spawn_box = spawn_box
        self.spawn_max_velocity = spawn_max_velocity
        self.rng = rng
        self.id = self.body.id
        
    @property
    def position(self) -> np.ndarray:
        return self.body.xpos
    
    @position.setter
    def position(self, value: np.ndarray) -> None:
        self.body.xpos = value
    
    @property
    def velocity(self) -> np.ndarray:
        return self.body.cvel[:3]
    
    @velocity.setter
    def velocity(self, value: np.ndarray) -> None:
        self.body.cvel[:3] = value
    
    @property
    def angular_velocity(self) -> np.ndarray:
        return self.body.qvel[3:6]
    
    @angular_velocity.setter
    def angular_velocity(self, value: np.ndarray) -> None:
        self.body.qvel[3:6] = value
    
    @property
    def imu_accel(self) -> np.ndarray:
        return self.data.sensor('imu_accel').data
    
    @property
    def imu_gyro(self) -> np.ndarray:
        return self.data.sensor('imu_gyro').data
    
    @property
    def imu_orientation(self) -> np.ndarray:
        return self.data.sensor('imu_orientation').data
    
    def reset(self):
        """
        Reset the drone's position, orientation, velocity, and angular velocity
        :return:
        """
        self.position = self.spawn_box[0] + (self.spawn_box[1] - self.spawn_box[0]) * self.rng.random(3)
        self.velocity = self.spawn_max_velocity * self.rng.random(3)


class Target:
    def __init__(self,
                 data: MjData,
                 spawn_box: np.ndarray,
                 spawn_max_velocity: float,
                 spawn_max_angular_velocity: float,
                 rng: np.random.Generator = default_rng()
                 ):
        """
        Initialize the Target object
        :param data:
        :param spawn_box:
        :param spawn_max_velocity:
        :param spawn_max_angular_velocity:
        """
        self.data = data
        self.body: _MjDataBodyViews = data.body('target')
        self.spawn_box = spawn_box
        self.spawn_max_velocity = spawn_max_velocity
        self.spawn_max_angular_velocity = spawn_max_angular_velocity
        self.rng = rng
        self.id = self.body.id
    
    @property
    def position(self) -> np.ndarray:
        return self.body.xpos
    
    @position.setter
    def position(self, value: np.ndarray) -> None:
        self.body.xpos = value
    
    @property
    def velocity(self) -> np.ndarray:
        return self.body.cvel[:3]
    
    @velocity.setter
    def velocity(self, value: np.ndarray) -> None:
        self.body.cvel[:3] = value
    
    @property
    def orientation(self) -> np.ndarray:
        return self.body.qpos[3:7]
    
    @orientation.setter
    def orientation(self, value: np.ndarray) -> None:
        self.body.qpos[3:7] = value
    
    def reset(self):
        """
        Reset the target's position, orientation, velocity, and angular velocity
        :return:
        """
        self.position = self.spawn_box[0] + (self.spawn_box[1] - self.spawn_box[0]) * self.rng.random(3)
        self.velocity = self.spawn_max_velocity * self.rng.random(3)


# region BaseMujocoEnv
class _BaseDroneEnv(MujocoEnv, EzPickle, ABC):
    """
    CustomMujocoEnv is a custom environment that extends the
    MujocoEnv. It is designed to simulate a drone in a 3D space
    with a target. The drone's task is to reach the target.
    """
    
    # region Initialization
    def __init__(self, **kwargs):
        """
        Initialize the CustomMujocoEnv
        :key xml_path: Path to the XML file that describes the
        :key sim_rate: Simulation rate
        :key dt: Simulation timestep
        :key height: Height of the camera
        :key width: Width of the camera
        :key drone_spawn_box: Box in which the drone can spawn. First
            numpy array is the lower bound and the second numpy array is
            the upper bound.
        :key drone_spawn_angle_range: Range of angles in which the drone
            can spawn. First numpy array is the lower bound and the second
            numpy array is the upper bound.
        :key drone_spawn_max_velocity: Maximum velocity at which the drone
            can spawn.
        :key drone_spawn_max_angular_velocity: Maximum angular velocity at
            which the drone can spawn.
        :key target_spawn_box: Box in which the target can spawn. First
            numpy array is the lower bound and the second numpy array is
            the upper bound.
        """
        # region Initialize MujocoEnv
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": 500,
            # render_fps will be set to dt^-1 after the model loads
        }
        height = kwargs.get('height', 480)
        width = kwargs.get('width', 640)
        observation_space: Space[ObsType] = Dict({
            "imu_accel":
                Box(low=-np.inf, high=np.inf, shape=(3,)),
            "imu_gyro":
                Box(low=-np.inf, high=np.inf, shape=(4,)),
            "imu_orientation":
                Box(low=-np.inf, high=np.inf, shape=(3,)),
            "image_0":
                Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8),
            "image_1":
                Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8),
        })
        
        MujocoEnv.__init__(
            self,
            model_path=XML_PATH,
            frame_skip=kwargs.get('frame_skip', 1),
            observation_space=observation_space,
            height=height,
            width=width,
        )
        EzPickle.__init__(
            self,
            XML_PATH,
            kwargs.get('frame_skip', 1),
            kwargs.get('dt', self.dt),
            **kwargs,
        )
        # Set action space
        self.action_space: Space = (
            Box(low=-1.0, high=1.0, shape=(self.model.nu,)))
        
        # Set simulation timestep
        self.model.opt.timestep = kwargs.get('dt', self.dt)
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        # endregion
        
        # region Initialize Random Number Generator and Model/Data
        # Initialize random number generator
        self.rng = default_rng()
        
        # Initialize model and data attributes
        self.model: MjModel = self.model
        self.data: MjData = self.data
        # endregion
        
        # region Drone Parameters
        self.drone = Drone(
            data=self.data,
            spawn_box=kwargs.get('drone_spawn_box',
                                 np.array([np.array([-10, -10, 0.5]), np.array([10, 10, 0.5])])),
            spawn_max_velocity=kwargs.get('drone_spawn_max_velocity', 0.5),
            rng=self.rng
        )
        # endregion
        
        # region Target Parameters
        self.target = Target(
            data=self.data,
            spawn_box=kwargs.get('target_spawn_box',
                                 np.array([np.array([-10, -10, 0.5]), np.array([10, 10, 0.5])])),
            spawn_max_velocity=kwargs.get('target_spawn_max_velocity', 0.5),
            spawn_max_angular_velocity=kwargs.get('target_spawn_max_angular_velocity', 0.5),
            rng=self.rng
        )
        # endregion
    
    # endregion
    
    # region Properties
    @property
    def drone_accel(self) -> np.ndarray:
        return self.data.sensor('imu_accel').data
    
    @property
    def drone_gyro(self) -> np.ndarray:
        return self.data.sensor('imu_gyro').data
    
    @property
    def drone_orientation(self) -> np.ndarray:
        return self.data.sensor('imu_orientation').data
    
    @property
    def camera_0(self) -> np.ndarray:
        return self.mujoco_renderer.render('rgb_array', 0)
    
    @property
    def camera_1(self) -> np.ndarray:
        return self.mujoco_renderer.render('rgb_array', 1)
    
    @property
    def target_position(self) -> np.ndarray:
        return self.data.body('target').xpos
    
    @property
    def target_velocity(self) -> np.ndarray:
        return self.data.body('target').cvel[:3]
    
    @property
    def drone_target_vector(self) -> np.ndarray:
        return self.target_position - self.data.body('drone').xpos
    
    @property
    def drone_hit_ground(self) -> bool:
        drone_id = self.model.geom('drone').id
        floor_id = self.model.geom('floor').id
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if (contact.geom1 == drone_id and contact.geom2 == floor_id) or (
                    contact.geom1 == floor_id and contact.geom2 == drone_id):
                return True
        return False
    
    # endregion
    
    # region Methods
    def place_target(self,
                     target_pos: np.ndarray,
                     target_orientation: np.ndarray) -> None:
        self.data.body('target').qpos[:3] = target_pos
        self.data.body('target').qpos[3:] = target_orientation
    
    def move_target(self, target_pos: np.ndarray) -> None:
        self.data.body('target').xpos = target_pos
    
    def pre_simulation(self) -> None:
        pass
    
    def reset_model(self) -> ObsType:
        self.drone.reset()
        self.target.reset()
        return self.observation
    
    # endregion
    
    # region Step Logic
    def step(
            self, action: ActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.pre_simulation()
        self.do_simulation(action, self.frame_skip)
        return (self.observation, self.reward,
                self.truncated, self.done, self.metrics)
    
    @property
    def observation(self) -> ObsType:
        return {
            "drone_position": self.data.qpos[0:3],
            "drone_velocity": self.data.qvel[0:3],
            "drone_orientation": self.data.qpos[3:7],
            "drone_angular_velocity": self.data.qvel[3:6],
            "image_0": self.mujoco_renderer.render('rgb_array', 0),
            "image_1": self.mujoco_renderer.render('rgb_array', 1),
        }
    
    @property
    @abstractmethod
    def reward(self) -> SupportsFloat:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def done(self) -> bool:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def truncated(self) -> bool:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def metrics(self) -> dict[str, Any]:
        raise NotImplementedError
    
    # endregion


# endregion

# region Tests
class _TestDroneEnv(_BaseDroneEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @property
    def reward(self) -> SupportsFloat:
        return 0.0
    
    @property
    def done(self) -> bool:
        return False
    
    @property
    def truncated(self) -> bool:
        return False
    
    @property
    def metrics(self) -> dict[str, Any]:
        return {}


def test_initialization_with_default_parameters():
    env = _TestDroneEnv()
    assert env.dt == 0.002
    assert env.height == 480
    assert env.width == 640
    assert np.array_equal(env.drone.spawn_box, np.array([np.array([-10, -10, 0.5]), np.array([10, 10, 0.5])]))
    assert env.drone.spawn_max_velocity == 0.5
    assert np.array_equal(env.target.spawn_box, np.array([np.array([-10, -10, 0.5]), np.array([10, 10, 0.5])]))
    assert env.target.spawn_max_velocity == 0.5
    assert env.target.spawn_max_angular_velocity == 0.5


def test_initialization_with_custom_parameters():
    custom_drone_spawn_box = np.array([np.array([-5, -5, 0.5]), np.array([5, 5, 0.5])])
    custom_drone_spawn_angle_range = np.array(
        [np.array([-np.pi / 8, -np.pi / 8, -np.pi / 8]), np.array([np.pi / 8, np.pi / 8, np.pi / 8])])
    custom_target_spawn_box = np.array([np.array([-5, -5, 0.5]), np.array([5, 5, 0.5])])
    env = _TestDroneEnv(height=600, width=800, drone_spawn_box=custom_drone_spawn_box,
                        drone_spawn_angle_range=custom_drone_spawn_angle_range,
                        target_spawn_box=custom_target_spawn_box)
    assert env.height == 600
    assert env.width == 800
    assert np.array_equal(env.drone.spawn_box, custom_drone_spawn_box)
    assert np.array_equal(env.target.spawn_box, custom_target_spawn_box)


def test_reset_model():
    env = _TestDroneEnv()
    env.reset_model()
    assert env.drone.spawn_box[0][0] <= env.drone.position[0] <= env.drone.spawn_box[1][0]
    assert env.drone.spawn_box[0][1] <= env.drone.position[1] <= env.drone.spawn_box[1][1]
    assert env.drone.spawn_box[0][2] <= env.drone.position[2] <= env.drone.spawn_box[1][2]


def test_hit_ground():
    env = _TestDroneEnv()
    assert env.drone_hit_ground is False
    env.drone.position = np.array([0, 0, 0])
    mj_step(env.model, env.data)
    assert env.drone_hit_ground is True

# endregion
