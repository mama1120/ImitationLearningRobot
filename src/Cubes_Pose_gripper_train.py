import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from joblib import dump, load

# Define CubeStackDataset
class CubeStackDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.inputs = []
        self.outputs_position = []  # EEF position
        self.outputs_orientation = []  # EEF orientation
        self.outputs_gripper = []   # Gripper state
        self.load_data()

    def load_data(self):
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        print(f"Found {len(files)} JSON files in {self.data_dir}")
        for file in files:
            filepath = os.path.join(self.data_dir, file)
            with open(filepath, 'r') as f:
                data = json.load(f)
                if "action_label" not in data:
                    continue  # Skip files without an action label
                self.inputs.append(self.extract_features(data))
                self.outputs_position.append(data["next_robot_state"]["eef_position"])
                self.outputs_orientation.append(data["next_robot_state"]["eef_orientation"])
                self.outputs_gripper.append(int(data["next_gripper_state"]))  # Gripper state (bool -> int)

    def extract_features(self, data):
        # Extract robot state, cube positions, cube sizes, and action label as features
        features = []
        features.extend(data["robot_state"]["eef_position"])
        features.extend(data["robot_state"]["eef_orientation"])
        for cube in data["cube_positions"].values():
            features.extend(cube["position"])
            features.extend(cube["orientation"])
        features.extend(data["cube_sizes"].values())
        features.append(data["action_label"])  # Action label as part of the features
        features.append(data["stacking_cube"])  # Add stacking cube index/identifier
        return features

    def get_data(self):
        return (np.array(self.inputs), 
                np.array(self.outputs_position), 
                np.array(self.outputs_orientation), 
                np.array(self.outputs_gripper))

# Load dataset
data_dir = "./demos"  # Update this path if needed
dataset = CubeStackDataset(data_dir)
X, y_position, y_orientation, y_gripper = dataset.get_data()

# Normalize features and labels
input_scaler = StandardScaler()
output_scaler_position = StandardScaler()
output_scaler_orientation = StandardScaler()

# For gripper state, we don't scale it because it's binary (0 or 1)
X_scaled = input_scaler.fit_transform(X)
y_position_scaled = output_scaler_position.fit_transform(y_position)
y_orientation_scaled = output_scaler_orientation.fit_transform(y_orientation)

# Save scalers for later use
dump(input_scaler, 'input_scaler.pkl')
dump(output_scaler_position, 'output_scaler_position.pkl')
dump(output_scaler_orientation, 'output_scaler_orientation.pkl')

# Combine outputs for the model (EEF position, orientation, and gripper state)
y_combined = np.concatenate([y_position_scaled, y_orientation_scaled, y_gripper.reshape(-1, 1)], axis=1)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_combined, test_size=0.2, random_state=42)

# Define neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(8, activation='linear')  # 3 for EEF position + 4 for orientation + 1 for gripper state
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=110, batch_size=32)

# Save the trained model
model.save("robot_stacking_model_with_orientation_and_gripper.keras")

# Evaluate the model on the test set
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# Function to test the model
def test_model(model, test_input, input_scaler_path, output_scaler_position_path, output_scaler_orientation_path):
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
    # Load scalers
    input_scaler = load(input_scaler_path)
    output_scaler_position = load(output_scaler_position_path)
    output_scaler_orientation = load(output_scaler_orientation_path)

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

# Example usage of test_model
test_sample = X[0]  # Use the first sample as an example
print(f"Test Sample (original): {test_sample}")
predicted_position, predicted_orientation, predicted_gripper_state = test_model(
    model, test_sample, 'input_scaler.pkl', 'output_scaler_position.pkl', 'output_scaler_orientation.pkl'
)
print(f"Predicted Position: {predicted_position}")
print(f"Predicted Orientation: {predicted_orientation}")
print(f"Predicted Gripper State: {predicted_gripper_state}")
