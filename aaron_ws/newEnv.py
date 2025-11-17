# Import
from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Panda
from robosuite.models.grippers import gripper_factory
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject
from robosuite.models.objects import HookFrame
from robosuite.utils.mjcf_utils import new_joint
import mujoco
import mujoco.viewer
from robosuite.utils.mjcf_utils import array_to_string
import robosuite.utils.transform_utils as T

# Main
# World
world = MujocoWorldBase()

# Panda robot
mujoco_robot = Panda()
gripper = gripper_factory('PandaGripper')
mujoco_robot.add_gripper(gripper)

# Add robot to world model
mujoco_robot.set_base_xpos([0, 0, 0])
world.merge(mujoco_robot)

# Table
mujoco_arena = TableArena()
mujoco_arena.set_origin([0.8, 0, 0])
world.merge(mujoco_arena)

# Add object
sphere = BallObject(
    name="sphere",
    size=[0.04],
    rgba=[0, 0.5, 0.5, 1]).get_obj()
sphere.set('pos', '1.0 0 1.0')
world.worldbody.append(sphere)

# Add HookFrame
hook = HookFrame(
    name="hook",
    frame_length=0.3,
    frame_height=0.2,
    frame_thickness=0.025,
    hook_height=0.05,  # Add a hook at the end
    grip_location=0.1,  # Add a grip for grasping
    grip_size=(0.02, 0.03),  # Grip radius and half-height
    tip_size=(0.02, 0.01, 0.015, 0.03),  # Optional cone tip
    use_texture=True,
)

# Get the object and merge it
hook_obj = hook.get_obj()
hook_obj.set('pos', '0.9 0.1 0.82')
hook_obj.set('quat', array_to_string(T.convert_quat(hook.init_quat, to="wxyz")))

# # Use init_quat to make it lie flat on the table
# import robosuite.utils.transform_utils as T
# quat_wxyz = T.convert_quat(hook.init_quat, to="wxyz")  # Convert to wxyz format for MuJoCo
# hook_obj.set('quat', f'{quat_wxyz[0]} {quat_wxyz[1]} {quat_wxyz[2]} {quat_wxyz[3]}')

# Merge materials and add to world
world.merge_assets(hook)
world.worldbody.append(hook_obj)

# Load MuJoCo model
model = world.get_model(mode="mujoco")
data = mujoco.MjData(model)

# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and data.time < 100:
        mujoco.mj_step(model, data)
        viewer.sync()

print("Done")