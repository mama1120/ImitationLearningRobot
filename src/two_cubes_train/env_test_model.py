import time
import os
import numpy as np
import pybullet as p
from pybullet_utils.bullet_client import BulletClient
from bullet_env.bullet_robot import BulletRobot, BulletGripper
from transform import Affine
from tensorflow.keras.models import load_model
import tensorflow as tf
from joblib import dump, load

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

# Load the trained model
model = load_model("robot_stacking_model_with_orientation_and_gripper.keras")

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
    - input_scaler_path: Path to the saved input scaler.
    - output_scaler_position_path: Path to the saved output scaler for EEF position.
    - output_scaler_orientation_path: Path to the saved output scaler for EEF orientation.

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


# Picking and stacking cubes
first_cube_position = None
stack_offset = 0.3
sleep = False

bullet_client = BulletClient(connection_mode=p.GUI)
bullet_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
if not RENDER:
    bullet_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
robot = BulletRobot(bullet_client=bullet_client, urdf_path="/home/jovyan/workspace/assets/urdf/robot.urdf")
gripper = BulletGripper(bullet_client=bullet_client, robot_id=robot.robot_id)
robot.home()

#Set the enviroment
env = BulletEnvironment(bullet_client, robot)


# Random cube positions and sizes
CUBE_POSITIONS = [[np.random.uniform(0.4, 0.9), np.random.uniform(-0.3, 0.3), 0.05] for _ in range(2)]
CUBE_SIZES = [0.08 - i * 0.01 for i in range(2)]
print("Cube sizes:", CUBE_SIZES)
CUBE_COLORS = ["1 0 0 1", "0 1 0 1"]
# Temporary directory for URDF files
temp_dir = "temp_urdf"
os.makedirs(temp_dir, exist_ok=True)

# Create cubes in random positions
cube_ids = []
for i, (position, size, color) in enumerate(zip(CUBE_POSITIONS, CUBE_SIZES, CUBE_COLORS)):
    urdf_path = os.path.join(temp_dir, f"cube_{i}.urdf")
    with open(urdf_path, "w") as f:
        f.write(URDF_TEMPLATE.format(size=size, mass=size * 3, color=color))
    cube_id = bullet_client.loadURDF(urdf_path, position, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    cube_ids.append(cube_id)
    env.add_object(cube_id, f"cube_{i}", size)  # Save cube size here

'''
# Define cube positions and orientations
cube_positions = {
    "cube_0": {
        "position": [0.8991902063150211, 0.03281084838416639, 0.027489221746996795],
        "orientation": [1.6014625499936277e-06, -1.2198582202730507e-05, -9.71492406165834e-07,  0.999999999923843]
    },
    "cube_1": {
        "position": [0.5469685036952939, -0.09430822815750642, 0.02238897540251399],
        "orientation": [0.0004563887982529381, -0.0009144727757633213, -7.167152757830409e-06,0.9999994776985831]
    }
}

cube_ids = []
for i, ( size, color) in enumerate(zip(CUBE_SIZES, CUBE_COLORS)):
    urdf_path = os.path.join(temp_dir, f"cube_{i}.urdf")
    with open(urdf_path, "w") as f:
        f.write(URDF_TEMPLATE.format(size=size, mass=size * 3, color=color))
    position = cube_positions[f'cube_{i}']['position']
    cube_id = bullet_client.loadURDF(urdf_path, position, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    cube_ids.append(cube_id)
    env.add_object(cube_id, f"cube_{i}", size)  # Save cube size here
'''
# Simulate the scene to settle objects
for _ in range(100):
    bullet_client.stepSimulation()
    time.sleep(1 / 100)

first_cube_position = None
for i in range(7):
    # Get cube position
    position, quat = bullet_client.getBasePositionAndOrientation(cube_id)
    cube_pose = Affine(position, quat)

    if first_cube_position is None:
        # Use the first cube as the base
        robot.home()
        gripper.open()
        first_cube_position = position
        print(f"First cube position (base): {first_cube_position}")
        continue  # Skip the first cube


    

    #get the state of the enviroment with the functions: eef pose, cube positions, cube sizes
    #This will be the input of the model
    # Get the state of the environment with the functions: eef pose, cube positions, cube sizes
    eef_pose = robot.get_eef_pose()
    print("EEF pose:", eef_pose)
    cube_positions = env.get_cube_positions()
    #print("Cube positions:", cube_positions)
    cube_sizes = env.get_cube_sizes()
    #print("Cube sizes:", cube_sizes)

    # Prepare the input for the model
    sample_input = [
        # EEF position
        eef_pose.translation[0], eef_pose.translation[1], eef_pose.translation[2],
        # EEF orientation (quaternion)
        eef_pose.quat[0], eef_pose.quat[1], eef_pose.quat[2], eef_pose.quat[3],
        # Cube 0 position
        cube_positions['cube_0']['position'][0], cube_positions['cube_0']['position'][1], cube_positions['cube_0']['position'][2],
        # Cube 0 orientation (quaternion)
        cube_positions['cube_0']['orientation'][0], cube_positions['cube_0']['orientation'][1], cube_positions['cube_0']['orientation'][2], cube_positions['cube_0']['orientation'][3],
        # Cube 1 position
        cube_positions['cube_1']['position'][0], cube_positions['cube_1']['position'][1], cube_positions['cube_1']['position'][2],
        # Cube 1 orientation (quaternion)
        cube_positions['cube_1']['orientation'][0], cube_positions['cube_1']['orientation'][1], cube_positions['cube_1']['orientation'][2], cube_positions['cube_1']['orientation'][3],
        # Cube sizes
        cube_sizes['cube_0'], cube_sizes['cube_1'], (i-1) # Include an action label
    ]
    print("Action State: ", i-1)

    # Predict the output using the model
    predicted_position, predicted_orientation, predicted_gripper = test_model(model, sample_input)
    predicted_orientation = np.squeeze(predicted_orientation)  # Removes dimensions of size 1
    gripper_state = predicted_gripper
    print("Gripper state:", gripper_state)    
    pred_target_pose = Affine(predicted_position, predicted_orientation)



    # Implement grasping the object
    #gripper_rotation = Affine(rotation=[0, np.pi, 0])
    print("Predicted target pose:", pred_target_pose)
    print("cube positions: ", cube_positions['cube_1']['position'][0], cube_positions['cube_1']['position'][1], cube_positions['cube_1']['position'][2])
    #pred_target_pose = pred_target_pose * gripper_rotation
    #print("Predicted target pose after rotation:", pred_target_pose)

    # Move to predicted pose
    robot.ptp(pred_target_pose)

    

    eef_pose = robot.get_eef_pose()
    #print("EEF pose after ptp:", eef_pose)
    #gripper.open()

    # Set the gripper state based on the prediction
    if gripper_state == 0:
       gripper.close()
    else:
        gripper.open()
    
    time.sleep(0)
    i = i + 1