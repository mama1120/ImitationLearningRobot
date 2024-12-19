import time
import os
import numpy as np
import pybullet as p
from pybullet_utils.bullet_client import BulletClient
from bullet_env.bullet_robot import BulletRobot, BulletGripper
from transform import Affine

# Setup
RENDER = True
URDF_TEMPLATE = """<?xml version="1.0" ?>
<robot name="cube">
    <material name="color">
        <color rgba="{color}"/>
    </material>
    <link name="baseLink">
        <contact>
            <lateral_friction value="3"/>
            <rolling_friction value="0.001"/>
            <inertia_scaling value="0.8"/>
        </contact>
        <inertial>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <mass value="{mass}"/>
            <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
        </inertial>
        <visual>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <geometry>
                <box size="{size} {size} {size}"/>
            </geometry>
            <material name="color"/>
        </visual>
        <collision>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <geometry>
                <box size="{size} {size} {size}"/>
            </geometry>
        </collision>
    </link>
</robot>
"""

bullet_client = BulletClient(connection_mode=p.GUI)
bullet_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
if not RENDER:
    bullet_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

bullet_client.resetSimulation()

robot = BulletRobot(bullet_client=bullet_client, urdf_path="/home/jovyan/workspace/assets/urdf/robot.urdf")
gripper = BulletGripper(bullet_client=bullet_client, robot_id=robot.robot_id)
robot.home()
home_pose = robot.get_eef_pose()

# Random cube positions and sizes
CUBE_POSITIONS = [[np.random.uniform(0.4, 0.9), np.random.uniform(-0.3, 0.3), 0.05] for _ in range(6)]
CUBE_SIZES = [0.08 - i * 0.01 for i in range(6)]
print("Cube sizes:", CUBE_SIZES)
CUBE_COLORS = ["1 0 0 1", "0 1 0 1", "0 0 1 1", "1 1 0 1", "1 0 1 1", "1 1 1 1"]

# Temporary directory for URDF files
temp_dir = "temp_urdf"
os.makedirs(temp_dir, exist_ok=True)

cube_ids = []
for i, (position, size, color) in enumerate(zip(CUBE_POSITIONS, CUBE_SIZES, CUBE_COLORS)):
    urdf_path = os.path.join(temp_dir, f"cube_{i}.urdf")
    with open(urdf_path, "w") as f:
        f.write(URDF_TEMPLATE.format(size=size, mass=size * 3, color=color))
    cube_id = bullet_client.loadURDF(urdf_path, position, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    cube_ids.append(cube_id)

# Simulate the scene to settle objects
for _ in range(100):
    bullet_client.stepSimulation()
    time.sleep(1 / 100)

# Picking and stacking cubes
first_cube_position = None
stack_offset = 0.3
sleep = False

for i, cube_id in enumerate(cube_ids):
    # Get cube position
    position, quat = bullet_client.getBasePositionAndOrientation(cube_id)
    cube_pose = Affine(position, quat)

    if first_cube_position is None:
        # Use the first cube as the base
        first_cube_position = position
        print(f"First cube position (base): {first_cube_position}")
        continue  # Skip the first cube

    # Move to pre-grasp position
    gripper_rotation = Affine(rotation=[0, np.pi, 0])
    target_pose = cube_pose * gripper_rotation
    pre_grasp_offset = Affine(translation=[0, 0, -0.35])
    pre_grasp_pose = target_pose * pre_grasp_offset
    robot.ptp(pre_grasp_pose)
    gripper.open()
    print("Pre-grasp pose:", pre_grasp_pose)
    if sleep:
        time.sleep(2)

    # Grasp cube
    robot.lin(target_pose)
    gripper.close()
    print("Grasped pose:", target_pose)
    if sleep:
        time.sleep(2)

    # Move up to avoid collisions
    lift_pose = target_pose * Affine(translation=[0, 0, -0.2])
    robot.lin(lift_pose)
    print("Lift pose:", lift_pose)
    if sleep:
        time.sleep(2)

    # Move to stacking position
    stack_position = list(first_cube_position)
    stack_position[2] += 0.05 + CUBE_SIZES[0] / 2  # Base cube height
    stack_position[2] += sum(CUBE_SIZES[1:i])  # Add height for stacked cubes

    stack_target = Affine(translation=stack_position, rotation=[0, np.pi, 0])

    # Approach above stacking position
    above_stack = stack_target * Affine(translation=[0, 0, -0.2])
    robot.lin(above_stack)
    print("Above stack position:", above_stack)
    if sleep:
        time.sleep(5)

    # Descend to stack position
    robot.lin(stack_target)
    if sleep:
        time.sleep(2)
    gripper.open()

    print("Stacked pose:", stack_target)
    if sleep:
        time.sleep(2)

    # Move back to home
    robot.ptp(home_pose)
    print("Home pose:", home_pose)
    if sleep:
        time.sleep(2)

# Clean up temporary URDF files
for filename in os.listdir(temp_dir):
    os.remove(os.path.join(temp_dir, filename))
os.rmdir(temp_dir)

bullet_client.disconnect()
