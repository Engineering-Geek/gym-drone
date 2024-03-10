import mujoco
import numpy as np

# Load the model and create data
model = mujoco.MjModel.from_xml_path('assets/UAV/scene.xml')
data = mujoco.MjData(model)

# Get the body ID for the target body
body_id = model.body('target').id

# Define the coefficients of the parabola: y = a*x^2 + b*x + c
a, b, c = 1, 0, 0

# Simulation loop
for t in range(100):
    x = 0.1 * t  # For example, linearly increasing x over time
    y = a * x ** 2 + b * x + c  # Calculate the corresponding y on the parabola
    
    # Update the body's position in qpos
    # Assuming the first 3 components of qpos for the body's
    # free joint correspond to its x, y, z positions0
    data.qpos[0] = x
    data.qpos[0 + 1] = y
    data.qpos[0 + 2] = 0  # Assuming z remains constant
    
    # Step the simulation
    mujoco.mj_step(model, data)
    print(data.qpos[0:3])  # Print the body's position
