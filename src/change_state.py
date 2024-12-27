import os
import json

# Paths
folder_path = "new_demos"  # Replace with the actual folder path

# Target modifications
pre_grasp_robot_state = {
    "eef_position": [
        0.6913298964500427,
        0.1742745339870453,
        0.4565303921699524
    ],
    "eef_orientation": [
        0.7071565638830705,
        -0.7070569645181409,
        -0.0001979143341199308,
        -6.255715782012504e-05
    ]
}

home_next_robot_state = {
    "eef_position": [
        0.6913298964500427,
        0.1742745339870453,
        0.4565303921699524
    ],
    "eef_orientation": [
        0.7071565638830705,
        -0.7070569645181409,
        -0.0001979143341199308,
        -6.255715782012504e-05
    ]
}

# Function to modify JSON files
def modify_json_files(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        # Process only JSON files
        if not file_name.endswith(".json"):
            continue
        
        # Load the file
        with open(file_path, "r") as file:
            data = json.load(file)

        # Modify for "pre_grasp" or action_label 0
        if "pre_grasp" in file_name or data.get("action_label") == 0:
            data["robot_state"] = pre_grasp_robot_state

        # Modify for "home" or action_label 5
        if "home" in file_name or data.get("action_label") == 5:
            data["next_robot_state"] = home_next_robot_state

        # Save the modified file
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)

# Modify the JSON files in the folder
modify_json_files(folder_path)
