import mujoco
import mujoco.viewer
import numpy as np
from mujoco._structs import _MjDataBodyViews

# Load the model and create the data object
m = mujoco.MjModel.from_xml_path('assets/UAV/scene.xml')
d = mujoco.MjData(m)

# Get the body ID for the target body
body: _MjDataBodyViews = d.body('target')


force = np.array([10000, 10000, 0])  # Force in x, y, z directions
torque = np.array([100, 100, 100])  # Torque in x, y, z directions
body.xfrc_applied[:3] = force
body.xfrc_applied[3:] = torque

i = 0
with mujoco.viewer.launch_passive(m, d) as v:
    while v.is_running():
        # if i < 20:
        #     i += 1
        #     body.xfrc_applied[:3] = force
        #     body.xfrc_applied[3:] = torque
            
        # Apply the force to the target body

        # Step the simulation
        mujoco.mj_step(m, d)

        # Print the position of the target body
        v.sync()
        
