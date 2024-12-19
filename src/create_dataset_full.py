import time
import os
import numpy as np
import pybullet as p
from pybullet_utils.bullet_client import BulletClient
from bullet_env.bullet_robot import BulletRobot, BulletGripper
from transform import Affine
import json

# Define some action labels:
ACTIONS = {
    "move_to_pre_grasp": 0,
    "move_to_grasp": 1,
    "grasp_cube": 2,
    "lift_cube": 3,
    "move_to_stack_position": 4,
    "stack_cube": 5,
    "return_home": 6
}

class BulletEnvironment:
    def __init__(self, bullet_client, robot):
        self.bullet_client = bullet_client
        self.robot = robot
        self.objects = {}  # Dictionary to store object IDs and their names
        self.gripper_state = False  # Initialize gripper state (False means closed, True means open)

    def add_object(self, object_id, name, size):
        """Register an object with its ID, name, and size."""
        self.objects[object_id] = {"name": name, "size": size}

    def get_cube_positions(self):
        """Get positions of all registered cubes."""
        cube_positions = {}
        for object_id, obj in self.objects.items():
            pos, ori = self.bullet_client.getBasePositionAndOrientation(object_id)
            cube_positions[obj["name"]] = {"position": pos, "orientation": ori}
        return cube_positions

    def get_robot_state(self):
        """Get the robot's end-effector pose."""
        eef_pose = self.robot.get_eef_pose()
        return {
            "eef_position": eef_pose.translation.tolist(),
            "eef_orientation": eef_pose.quat.tolist(),
        }

    def get_gripper_state(self):
        """Get the gripper's open/close state."""
        return self.gripper_state

    def set_gripper_state(self, state):
        """Manually set the gripper state (open/close)."""
        self.gripper_state = state

    def get_cube_sizes(self):
        """Get the sizes of all cubes."""
        return {obj["name"]: obj["size"] for obj in self.objects.values()}

    def save_demonstration(self, filename, action_label):
        """Save the current state of the environment and the action to a file."""
        data = {
            "action_label": action_label,
            "robot_state": self.get_robot_state(),
            "cube_positions": self.get_cube_positions(),
            "gripper_state": self.get_gripper_state(),
            "cube_sizes": self.get_cube_sizes(),
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Demonstration saved to {filename}")

def stack_cubes(bullet_client, robot, gripper, urdf_template, cube_positions, cube_sizes, cube_colors, env, dataset):
    """Stacks cubes and saves the demonstration with action labels."""
    temp_dir = "temp_urdf"
    os.makedirs(temp_dir, exist_ok=True)

    cube_ids = []
    for i, (position, size, color) in enumerate(zip(cube_positions, cube_sizes, cube_colors)):
        urdf_path = os.path.join(temp_dir, f"cube_{i}.urdf")
        with open(urdf_path, "w") as f:
            f.write(urdf_template.format(size=size, mass=size * 3, color=color))
        cube_id = bullet_client.loadURDF(urdf_path, position, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        cube_ids.append(cube_id)
        env.add_object(cube_id, f"cube_{i}", size)  # Save cube size here

    for _ in range(100):
        bullet_client.stepSimulation()
        time.sleep(1 / 100)

    first_cube_position = None
    for i, cube_id in enumerate(cube_ids):
        position, quat = bullet_client.getBasePositionAndOrientation(cube_id)
        cube_pose = Affine(position, quat)

        if first_cube_position is None:
            first_cube_position = position
            continue

        # Pick up the cube (pre-grasp, grasp, lift)
        gripper_rotation = Affine(rotation=[0, np.pi, 0])
        target_pose = cube_pose * gripper_rotation
        pre_grasp_offset = Affine(translation=[0, 0, -0.35])
        pre_grasp_pose = target_pose * pre_grasp_offset
        robot.ptp(pre_grasp_pose)
        gripper.open()
        env.set_gripper_state(True)  # Update gripper state to open

        # Save pre-grasp action
        env.save_demonstration(f"demo_{len(dataset)}_0pre_grasp.json", ACTIONS["move_to_pre_grasp"])

        # Move to grasp position
        robot.lin(target_pose)
        env.save_demonstration(f"demo_{len(dataset)}_1move_to_grasp.json", ACTIONS["move_to_grasp"])

        #Grasp the cube
        gripper.close()
        env.set_gripper_state(False)  # Update gripper state to closed
        # Save grasp action
        env.save_demonstration(f"demo_{len(dataset)}_2grasp.json", ACTIONS["grasp_cube"])

        # Move up to avoid collisions
        lift_pose = target_pose * Affine(translation=[0, 0, -0.2])
        robot.lin(lift_pose)
        # Save lift action
        env.save_demonstration(f"demo_{len(dataset)}_3lift.json", ACTIONS["lift_cube"])

        # Move to stacking position
        stack_position = list(first_cube_position)
        stack_position[2] += 0.05 + cube_sizes[0] / 2
        stack_position[2] += sum(cube_sizes[1:i])

        # Approach above stacking position
        stack_target = Affine(translation=stack_position, rotation=[0, np.pi, 0])
        above_stack = stack_target * Affine(translation=[0, 0, -0.2])
        robot.lin(above_stack)
        # Save stacking action
        env.save_demonstration(f"demo_{len(dataset)}_4move_to_stack.json", ACTIONS["move_to_stack_position"])

        # Descend to stack position
        robot.lin(stack_target)
        gripper.open()
        env.set_gripper_state(True)  # Update gripper state to open

        # Save stack action
        env.save_demonstration(f"demo_{len(dataset)}_5stack.json", ACTIONS["stack_cube"])

        # Go to the home position, to avoid collisions
        home_pose = robot.get_eef_pose()
        robot.ptp(home_pose)

        # Save return home action
        env.save_demonstration(f"demo_{len(dataset)}_6home.json", ACTIONS["return_home"])

    # Clean up temporary URDF files
    for filename in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, filename))
    os.rmdir(temp_dir)

    success = input("Was the stacking successful? (y/n): ").strip().lower() == "y"
    return success

def main():
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

    dataset = []

    while True:
        bullet_client.resetSimulation()

        robot = BulletRobot(bullet_client=bullet_client, urdf_path="/home/jovyan/workspace/assets/urdf/robot.urdf")
        gripper = BulletGripper(bullet_client=bullet_client, robot_id=robot.robot_id)
        robot.home()

        env = BulletEnvironment(bullet_client, robot)

        # Generate positions, sizes, and colors for the cubes
        cube_positions = [[np.random.uniform(0.4, 0.9), np.random.uniform(-0.3, 0.3), 0.05] for _ in range(6)]
        cube_sizes = [0.08 - i * 0.01 for i in range(6)]
        cube_colors = ["1 0 0 1", "0 1 0 1", "0 0 1 1", "1 1 0 1", "1 0 1 1", "1 1 1 1"]
        #cube_positions = [[np.random.uniform(0.4, 0.9), np.random.uniform(-0.3, 0.3), 0.05] for _ in range(2)]
        #cube_sizes = [0.08 - i * 0.01 for i in range(2)]
        #cube_colors = ["1 0 0 1", "0 1 0 1"]
        
        # Stack cubes and save the demonstration
        success = stack_cubes(bullet_client, robot, gripper, URDF_TEMPLATE, cube_positions, cube_sizes, cube_colors, env, dataset)

        if success:
            # Add the demonstration to the dataset
            dataset.append((cube_positions, cube_sizes, cube_colors))
            env.save_demonstration(f"demo_{len(dataset)}.json", ACTIONS["return_home"])  # Add an appropriate action label
            print("Scene saved to dataset.")
        else:
            print("Scene discarded.")

        continue_process = input("Do you want to stack again? (y/n): ").strip().lower()
        if continue_process != "y":
            break


if __name__ == "__main__":
    main()