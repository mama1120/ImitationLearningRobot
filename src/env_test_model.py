import time
import os
import numpy as np
import pybullet as p
from pybullet_utils.bullet_client import BulletClient
from bullet_env.bullet_robot import BulletRobot, BulletGripper
from transform import Affine
from tensorflow.keras.models import load_model
import tensorflow as tf
from joblib import load


# Setup
RENDER = True

# Load the trained model
model = load_model("BC.keras")

# Load the saved scalers
input_scaler = load("input_scaler.pkl")
output_scaler_position = load("output_scaler_position.pkl")
output_scaler_orientation = load("output_scaler_orientation.pkl")

# Function to test the model
def test_model(model, test_input):
    """
    Test the trained model with a given input.

    Args:
    - model: The trained Keras model.
    - test_input: A numpy array of shape (1, X.shape[1]) representing a single test sample.

    Returns:
    - predicted_position: The predicted EEF position in the original range.
    - predicted_orientation: The predicted EEF orientation in the original range.
    - predicted_gripper_state: The predicted gripper state as a binary value.
    """
    try:
        # Scale input
        test_input_scaled = input_scaler.transform(np.array(test_input).reshape(1, -1))
        # Get prediction in scaled space
        predicted_output_scaled = model.predict(test_input_scaled)

        # Split predictions
        predicted_position_scaled = predicted_output_scaled[0, :3]
        predicted_orientation_scaled = predicted_output_scaled[0, 3:7]
        predicted_gripper_state = predicted_output_scaled[0, 7]

        # Inverse scale EEF position and orientation
        predicted_position = output_scaler_position.inverse_transform(predicted_position_scaled.reshape(1, -1))
        predicted_orientation = output_scaler_orientation.inverse_transform(predicted_orientation_scaled.reshape(1, -1))

        # Convert gripper state to binary (0 or 1)
        predicted_gripper_state = int(round(predicted_gripper_state))

        return predicted_position, predicted_orientation, predicted_gripper_state
    except Exception as e:
        raise ValueError(f"Error in test_model: {e}")

# Define the Bullet environment and its functions to interact with the simulation and get the current state
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

    def get_cube_sizes(self):
        """Get the sizes of all cubes."""
        return {obj["name"]: obj["size"] for obj in self.objects.values()}

class Expert:
    def __init__(self):
        self.position = None
        self.orientation = None
        self.gripper_state = None
        self.linptp = None

    def expert_policy(self, sample_input):
        stacking_cube = int(sample_input[-1])  # Stacking cube index
        action = int(sample_input[-2])  # Action label
        cube_sizes = sample_input[42:47]

        # Extract the position and orientation of the stacking cube
        stacking_cube_arr_index = 7 + stacking_cube * 7
        stacking_cube_position = sample_input[stacking_cube_arr_index:stacking_cube_arr_index + 3]
        stacking_cube_orientation = sample_input[stacking_cube_arr_index + 3:stacking_cube_arr_index + 7]
        cube_pose = Affine(stacking_cube_position, stacking_cube_orientation)

        # Get the position of the base stack (largest cube) and add the height of the other cubes
        base_cube_position = sample_input[7:10]
        base_cube_position[2] += 0.05 + cube_sizes[0] / 2
        base_cube_position[2] += sum(cube_sizes[1:stacking_cube-1])

        match action:
            case 0:  # Move to pre-grasp position
                gripper_rotation = Affine(rotation=[0, np.pi, 0])
                target_pose = cube_pose * gripper_rotation
                pre_grasp_offset = Affine(translation=[0, 0, -0.35])
                pre_grasp_pose = target_pose * pre_grasp_offset
                self.position = pre_grasp_pose.translation
                self.orientation = pre_grasp_pose.rotation
                self.gripper_state = 1
                self.linptp = 0

            case 1:  # Move to grasp
                gripper_rotation = Affine(rotation=[0, np.pi, 0])
                target_pose = cube_pose * gripper_rotation
                self.position = target_pose.translation
                self.orientation = target_pose.rotation
                self.gripper_state = 0
                self.linptp = 1



            case 2:  # Lift cube
                gripper_rotation = Affine(rotation=[0, np.pi, 0])
                target_pose = cube_pose * gripper_rotation
                lift_pose = target_pose * Affine(translation=[0, 0, -0.2])
                self.position = lift_pose.translation
                self.orientation = lift_pose.rotation
                self.gripper_state = 0
                self.linptp = 1



            case 3:  # Move to stack position
                stack_position = list(base_cube_position)
                stack_position[2] += 0.05 + sample_input[-3] / 2  # Add base cube height
                stack_target = Affine(translation=stack_position, rotation=[0, np.pi, 0])
                above_stack = stack_target * Affine(translation=[0, 0, -0.2])
                self.position = above_stack.translation
                self.orientation = above_stack.rotation
                self.gripper_state = 0
                self.linptp = 0


            case 4:  # Stack cube
                stack_position = list(base_cube_position)
                stack_position[2] += 0.05 + sample_input[-3] / 2  # Add base cube height
                stack_target = Affine(translation=stack_position, rotation=[0, np.pi, 0])
                self.position = stack_target.translation
                self.orientation = stack_target.rotation
                self.gripper_state = 1
                self.linptp = 1


            case 5:  # Return home
                self.position = [0.6913298964500427, 0.1742745339870453,0.4565303921699524]
                self.orientation = [0.7071565638830705, -0.7070569645181409,-0.0001979143341199308, -6.255715782012504e-05]
                self.gripper_state = 0
                self.linptp = 0


        return self.position, self.orientation, self.gripper_state,  self.linptp


# Connect to the PyBullet simulator
bullet_client = BulletClient(connection_mode=p.GUI)
bullet_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
if not RENDER:
    bullet_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
robot = BulletRobot(bullet_client=bullet_client, urdf_path="/home/jovyan/workspace/assets/urdf/robot.urdf")
gripper = BulletGripper(bullet_client=bullet_client, robot_id=robot.robot_id)
robot.home()

# Set the environment
env = BulletEnvironment(bullet_client, robot)

# Predefined cube parameters
CUBE_POSITIONS = [[np.random.uniform(0.4, 0.9), np.random.uniform(-0.3, 0.3), 0.05] for _ in range(5)]
CUBE_SIZES = [0.08 - i * 0.01 for i in range(5)]
CUBE_URDF_PATHS = [f"/home/jovyan/workspace/src/cubes_urdf/cube{i}.urdf" for i in range(5)]

# Load cubes from existing URDF files
cube_ids = []
for i, (position, size, urdf_path) in enumerate(zip(CUBE_POSITIONS, CUBE_SIZES, CUBE_URDF_PATHS)):
    cube_id = bullet_client.loadURDF(urdf_path, position, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    cube_ids.append(cube_id)
    env.add_object(cube_id, f"cube_{i}", size)  # Save cube size here

# Simulate the scene to settle objects
for _ in range(100):
    bullet_client.stepSimulation()
    time.sleep(1 / 100)

first_cube_position = None
expert = Expert()
for j, cube_id in enumerate(cube_ids):  # Loop through all cubes
    position, quat = bullet_client.getBasePositionAndOrientation(cube_id)
    cube_pose = Affine(position, quat)
    print("Cube pose:", cube_pose)

    # Skip first cube
    if first_cube_position is None:
        first_cube_position = position
        continue

    for i in range(6):  # Loop through all actions
        # Get cube position
        position, quat = bullet_client.getBasePositionAndOrientation(cube_id)
        cube_pose = Affine(position, quat)

        # Get the state of the environment with the functions: eef pose, cube positions, cube sizes
        eef_pose = robot.get_eef_pose()
        print("EEF pose:", eef_pose)
        cube_positions = env.get_cube_positions()
        cube_sizes = env.get_cube_sizes()

        sample_input = [
            # EEF position (3)
            eef_pose.translation[0], eef_pose.translation[1], eef_pose.translation[2],
            # EEF orientation (4)
            eef_pose.quat[0], eef_pose.quat[1], eef_pose.quat[2], eef_pose.quat[3],
        ] + [
            # Cube positions, orientations, and sizes for all cubes
            item
            for k in range(5)  # Adjust this to match the number of cubes used during training
            for item in (
                cube_positions[f'cube_{k}']['position'][0],
                cube_positions[f'cube_{k}']['position'][1],
                cube_positions[f'cube_{k}']['position'][2],
                cube_positions[f'cube_{k}']['orientation'][0],
                cube_positions[f'cube_{k}']['orientation'][1],
                cube_positions[f'cube_{k}']['orientation'][2],
                cube_positions[f'cube_{k}']['orientation'][3],
            )
        ] + [
            0.08, 0.07, 0.06, 0.05, 0.04  # Cube Sizes
        ] + [
            i,  # Action label (1 feature)
            j   # Stacking cube index (1 feature)
        ]

        # Input of the model
        print("Sample input:", sample_input)
        
        #Get the expert output and compare it to the models output
        expert_position, expert_orientation, expert_gripper, linptp = expert.expert_policy(sample_input)

        # Predict the output using the model
        predicted_position, predicted_orientation, predicted_gripper = test_model(model, sample_input)
        predicted_orientation = np.squeeze(predicted_orientation)  # Removes dimensions of size 1
        gripper_state = predicted_gripper

        #Compare Expert and Model
        print(f"Expert: Pos={expert_position}, Ori={expert_orientation}, Gripper={expert_gripper}")
        print(f"Model: Pos={predicted_position}, Ori={predicted_orientation}, Gripper={predicted_gripper}")

        pred_target_pose = Affine(predicted_position, predicted_orientation)
        expert_target_pose = Affine(expert_position, expert_orientation)

        # Move to predicted pose
        if linptp:
            robot.lin(pred_target_pose)
        else:
            robot.ptp(pred_target_pose)

        # Get the EEF pose after moving
        eef_pose = robot.get_eef_pose()
        print("EEF pose after ptp:", eef_pose)

        # Set the gripper state based on the prediction
        if gripper_state == 0:
            gripper.close()
        else:
            gripper.open()

        # Wait for 1 second
        time.sleep(0)
