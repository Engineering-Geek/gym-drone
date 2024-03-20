from gym_drone.utils.urdf2xml import urdf2xml
import os

path = os.path.dirname(__file__)
urdf2xml(os.path.join(path, "assets", "robot.urdf"), os.path.join(path, "drone2.xml"))

