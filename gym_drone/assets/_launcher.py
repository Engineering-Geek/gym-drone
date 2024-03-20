from mujoco import viewer, MjData, MjModel
import os

path = os.path.dirname(__file__)

model = MjModel.from_xml_path(os.path.join(path, "drone2.xml"))
data = MjData(model)

viewer.launch(model, data)

