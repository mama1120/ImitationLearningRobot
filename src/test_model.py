import numpy as np
from tensorflow.keras.models import load_model
from joblib import load  # For loading scalers saved with joblib

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


# Example test input (randomized; replace with actual test data)
# The input shape of the the model increases depending on the number of cubes
#The current model needs 49 features 
sample_input = [
np.float64(0.45029395818710327), np.float64(0.17526382207870483), np.float64(0.37393543124198914), np.float64(0.0015599028018414701), np.float64(0.9999987511156858), np.float64(-4.315399122195505e-06), np.float64(-0.00025387338317639274), 0.749313575116985, 0.18882261702656009, 0.027489221937301227, 1.6059899364362805e-06, -1.2196132416636609e-05, -9.706692545356987e-07, 0.9999999999238665, 0.4484947933576639, 0.17135565489172574, 0.02248939419468198, 1.2725118246012843e-06, -1.0836646961683303e-05, -7.952773391386721e-07, 0.9999999999401576, 0.5846415347179411, 0.014543836447435688, 0.017489364190160703, 1.833124975878135e-06, -1.3297601234027482e-05, -9.95382122776807e-07, 0.9999999999094114, 0.7004236206284927, -0.02249833335113238, 0.0124892388994016, 3.0731938653747394e-06, -1.9155055177314055e-05, -1.1870001394928645e-06, 0.9999999998111152, 0.5639918243448311, -0.12984489618240677, 0.007488088433710282, 5.616623177517707e-07, -5.915712616314598e-05, -3.408466972278792e-06, 0.9999999982442507, 0.08, 0.07, 0.06, 0.05, 0.04, 1, 1
]
print(f"Test Input: {sample_input}")
print(f"Test Input Shape: {len(sample_input)}")

# Test the model
try:
    predicted_position, predicted_orientation, predicted_gripper = test_model(model, sample_input)
    print(f"Predicted Position: {predicted_position}")
    print(f"Predicted Orientation: {predicted_orientation}")
    print(f"Predicted Gripper State: {predicted_gripper}")
except Exception as e:
    print(f"Error while testing the model: {e}")
