import pinocchio as pin
import numpy as np
import mujoco 

def printmodelparam(model):
    model = mujoco.MjModel.from_xml_path("./crazydog_urdf/urdf/test.xml")
    data= mujoco.MjData(model)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
    total_mass = model.body_subtreemass[body_id]
    print(f"total mass: {total_mass:.4f} kg")

    wheel_mass = model.body_mass[8]
    print(f"wheel_mass:{wheel_mass}[kg]")

    mujoco.mj_forward(model,data)
    pos_L = data.xpos[8]
    pos_R = data.xpos[5]
    dist_y = abs(pos_L[1] - pos_R[1])
    print(f"dist_y:{dist_y}[m]")

model = mujoco.MjModel.from_xml_path("./crazydog_urdf/urdf/test.xml")
printmodelparam(model)