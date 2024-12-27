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
    
    def check_cubes_on_table(self, table_height=0.0, threshold=0.03):
        """
        Check which cubes are still on the table and which are not.

        Args:
            table_height: The height of the table in the simulation (default: 0.0).
            threshold: Tolerance for determining if a cube is on the table (default: 0.01).

        Returns:
            on_table: List of object names that are still on the table.
            off_table: List of object names that are not on the table.
        """
        on_table = []
        off_table = []

        for object_id, obj in self.objects.items():
            # Get the position of the cube
            cube_position, _ = self.bullet_client.getBasePositionAndOrientation(object_id)
            cube_height = cube_position[2]  # Z-coordinate (height)

            # Check if the cube's height is within the threshold of the table height
            if abs(cube_height - table_height) <= threshold:
                on_table.append(obj["name"])
            else:
                off_table.append(obj["name"])

        return on_table, off_table

def stack_cubes(bullet_client, robot, gripper, urdf_path, cube_positions, cube_sizes, cube_colors, env, dataset):
    """Stacks cubes and saves the demonstration with action labels."""

    # Create cubes in random positions
    cube_ids = []
    for i, (position, size, urdf_path) in enumerate(zip(cube_positions, cube_sizes, urdf_path)):
        cube_id = bullet_client.loadURDF(urdf_path, position, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        cube_ids.append(cube_id)
        env.add_object(cube_id, f"cube_{i}", size)  # Save cube size here

    for _ in range(100):
        bullet_client.stepSimulation()
        time.sleep(1 / 100)

    # Skip the first cube and stack the rest
    first_cube_position = None
    for _ in range(len(cube_ids) - 1):  # Stack all but the first cube
        current_cube_attempt = 0
        stacking_success = False

        while not stacking_success:
            # Determine cubes on the table dynamically
            cubes_on_table, _ = env.check_cubes_on_table()
            print("Cubes on table:", cubes_on_table)
            print(f"cube_sizes on table: {type(cubes_on_table)}")
            print("Cube IDs:", cube_ids)
            cube_sizes = env.get_cube_sizes()
            print("Cube sizes:", cube_sizes)
            print(f"cube_sizes type: {type(cube_sizes)}")

            # Find the largest and second-largest cubes on the table
            cube_sizes_on_table = {cube: cube_sizes[cube] for cube in cubes_on_table}
            print("Cube sizes on table:", cube_sizes_on_table)
            largest_cube = max(cube_sizes_on_table, key=cube_sizes_on_table.get)
            second_largest_cube = max(
                (cube for cube in cube_sizes_on_table if cube != largest_cube),
                key=cube_sizes_on_table.get,
                default=None
            )

            if not second_largest_cube:
                print("No second-largest cube found. Stopping.")
                return False

            # Get the corresponding cube ID
            j = int(second_largest_cube.split('_')[-1])
            print("Cube index:", j)
            cube_id = cube_ids[j]

            for i in range(6):  # Loop through all actions
                # Get cube position
                position, quat = bullet_client.getBasePositionAndOrientation(cube_id)
                cube_pose = Affine(position, quat)

                # Get the state of the environment
                eef_pose = robot.get_eef_pose()
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
                    for k in range(5)
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

                # Predict the output using the model
                predicted_position, predicted_orientation, predicted_gripper = test_model(model, sample_input)
                predicted_orientation = np.squeeze(predicted_orientation)  # Removes dimensions of size 1
                gripper_state = predicted_gripper
                pred_target_pose = Affine(predicted_position, predicted_orientation)

                # Move to predicted pose
                robot.ptp(pred_target_pose)

                # Set the gripper state based on the prediction
                if gripper_state == 0:
                    gripper.close()
                    print("Gripper closed")
                else:
                    gripper.open()
                    print("Gripper opened")

                print("Action:", i)
                print("Stacking cube index:", j)

            # Check if the cube is correctly stacked
            cube_positions = env.get_cube_positions()
            position, _ = bullet_client.getBasePositionAndOrientation(cube_id)
            expected_height = sum([cube_sizes[f'cube_{i}'] for i in range(j)]) + cube_sizes[f'cube_{j}'] / 2
            tolerance = 0.02  # Adjust as needed
            if abs(position[2] - expected_height) <= tolerance:
                stacking_success = True
                current_cube_attempt = 0
                print(f"Cube {j} stacked successfully at height {position[2]} (expected {expected_height}). After {current_cube_attempt} attempts.")
                time.sleep(3)
            else:
                print(f"Cube {j} not stacked correctly. Current height: {position[2]}, Expected: {expected_height}. Retrying...")
                current_cube_attempt += 1
                print(f"Attempt {current_cube_attempt}")
                time.sleep(1)
                if current_cube_attempt >= 3:
                    print(f"Failed to stack cube {j} after {current_cube_attempt} attempts.")
                    return False

    return True


def main():
    RENDER = True
# Create a BulletClient and configure the visualizer
    bullet_client = BulletClient(connection_mode=p.GUI)
    bullet_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    if not RENDER:
        bullet_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

# Create an empty dataset to store the demonstrations
    dataset = []
    attempts = 0
    # Loop to create and save demonstrations
    while True:
        attempts += 1
        bullet_client.resetSimulation()

        robot = BulletRobot(bullet_client=bullet_client, urdf_path="/home/jovyan/workspace/assets/urdf/robot.urdf")
        gripper = BulletGripper(bullet_client=bullet_client, robot_id=robot.robot_id)
        robot.home()
        
        # Create a BulletEnvironment instance
        env = BulletEnvironment(bullet_client, robot)

        # Generate positions, sizes, and colors for the cubes
        cube_positions = [[np.random.uniform(0.4, 0.9), np.random.uniform(-0.3, 0.3), 0.05] for _ in range(5)]
        cube_sizes = [0.08 - i * 0.01 for i in range(5)]
        cube_colors = ["1 0 0 1", "0 1 0 1", "0 0 1 1", "1 1 0 1", "1 0 1 1"]
        CUBE_URDF_PATHS = [f"/home/jovyan/workspace/assets/urdf/cube{i}.urdf" for i in range(5)]

        success = stack_cubes(bullet_client, robot, gripper, CUBE_URDF_PATHS, cube_positions, cube_sizes, cube_colors, env, dataset)
        #Wait 5 seconds before starting a new scene, only for debugging
        #time.sleep(5)
        if success:
            print("Stacking successful. After attempts: ", attempts)
            time.sleep(8)
            break
        else:
            print("Stacking failed. Restarting scene.")
            print("Attempts: ", attempts)
            if(attempts > 6):
                break
if __name__ == "__main__":
    main()
