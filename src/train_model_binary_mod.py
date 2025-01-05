import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.callbacks import LearningRateScheduler
from joblib import dump, load
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Skipping file {file} due to JSON error: {e}")
                    continue
                if "action_label" not in data:
                    continue  # Skip files without an action label
                self.inputs.append(self.extract_features(data))
                self.outputs_position.append(data["next_robot_state"]["eef_position"])
                self.outputs_orientation.append(data["next_robot_state"]["eef_orientation"])
                self.outputs_gripper.append(int(data["next_gripper_state"]))  # Gripper state (bool -> int)

    def extract_features(self, data):
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

# Test the model
def test_model(model, test_input, input_scaler_path, output_scaler_position_path, output_scaler_orientation_path):
    input_scaler = load(input_scaler_path)
    output_scaler_position = load(output_scaler_position_path)
    output_scaler_orientation = load(output_scaler_orientation_path)

    test_input_scaled = input_scaler.transform(np.array(test_input).reshape(1, -1))
    predicted_position_orientation, predicted_gripper = model.predict(test_input_scaled)
    
    predicted_position = output_scaler_position.inverse_transform(predicted_position_orientation[:, :3])
    predicted_orientation = output_scaler_orientation.inverse_transform(predicted_position_orientation[:, 3:])
    predicted_gripper_state = int(round(predicted_gripper[0, 0]))

    return predicted_position, predicted_orientation, predicted_gripper_state

# Load dataset
data_dir = "./new_demos"  # Update this path if needed
dataset = CubeStackDataset(data_dir)
X, y_position, y_orientation, y_gripper = dataset.get_data()

# Normalize features and labels
input_scaler = StandardScaler()
output_scaler_position = StandardScaler()
output_scaler_orientation = StandardScaler()

X_scaled = input_scaler.fit_transform(X)
y_position_scaled = output_scaler_position.fit_transform(y_position)
y_orientation_scaled = output_scaler_orientation.fit_transform(y_orientation)

# Save scalers
dump(input_scaler, 'input_scaler.pkl')
dump(output_scaler_position, 'output_scaler_position.pkl')
dump(output_scaler_orientation, 'output_scaler_orientation.pkl')

# Combine outputs
y_combined = np.concatenate([y_position_scaled, y_orientation_scaled, y_gripper.reshape(-1, 1)], axis=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_combined, test_size=0.2, random_state=42)

# Define learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch < 50:
        return lr
    elif epoch < 100:
        return lr * 0.1
    else:
        return lr * 0.01

# Define model with batch normalization and dropout
input_layer = Input(shape=(X_train.shape[1],))
shared = Dense(256, activation='relu')(input_layer)
shared = BatchNormalization()(shared)
shared = Dropout(0.3)(shared)
shared = Dense(512, activation='relu')(shared)
shared = BatchNormalization()(shared)
shared = Dropout(0.3)(shared)
shared = Dense(256, activation='relu')(shared)
shared = BatchNormalization()(shared)

regression_output = Dense(7, activation='linear', name='regression_output')(shared)
classification_output = Dense(1, activation='sigmoid', name='classification_output')(shared)

model = Model(inputs=input_layer, outputs=[regression_output, classification_output])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={
        'regression_output': MeanSquaredError(),
        'classification_output': BinaryCrossentropy()
    },
    metrics={
        'regression_output': 'mae',
        'classification_output': 'accuracy'
    }
)

# Split labels
y_train_position_orientation = y_train[:, :7]
y_train_gripper = y_train[:, 7]
y_test_position_orientation = y_test[:, :7]
y_test_gripper = y_test[:, 7]

# Train the model
history = model.fit(
    X_train,
    {'regression_output': y_train_position_orientation, 'classification_output': y_train_gripper},
    validation_data=(
        X_test,
        {
            'regression_output': y_test_position_orientation,
            'classification_output': y_test_gripper
        }
    ),
    epochs=125,
    batch_size=32,
    callbacks=[LearningRateScheduler(lr_scheduler)],
    verbose=1
)

# Save the model
model.save("BC_Grip_Binary_Expanded.keras")

# Evaluate the model
loss, regression_loss, classification_loss, regression_mae, classification_accuracy = model.evaluate(
    X_test,
    {'regression_output': y_test_position_orientation, 'classification_output': y_test_gripper}
)

# Example usage
test_sample = X[0]
predicted_position, predicted_orientation, predicted_gripper_state = test_model(
    model, test_sample, 'input_scaler.pkl', 'output_scaler_position.pkl', 'output_scaler_orientation.pkl'
)
print(f"Test Sample: {test_sample}")
print(f"Predicted Position: {predicted_position}")
print(f"Predicted Orientation: {predicted_orientation}")
print(f"Predicted Gripper State: {predicted_gripper_state}")