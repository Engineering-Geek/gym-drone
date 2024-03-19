from mujoco import MjData, MjModel, mj_saveModel, mj_saveLastXML
import mujoco.viewer
from typing import Union
from pathlib import Path


def urdf2xml(urdf_file: Union[Path, str], xml_file: Union[Path, str]):
    model: MjModel = MjModel.from_xml_path(urdf_file)
    mj_saveLastXML(xml_file, model)


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(__file__))
    urdf2xml("../assets/assets/robot.urdf", "../gym_drone/assets/drone.xml")

