import os

# Define the folder containing your files
folder_path = "new_demos2"

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    # Check if the file name starts with "demo_"
    if filename.startswith("demo_") and filename.endswith(".json"):
        # Extract the demonstration number
        parts = filename.split("_")
        demo_number = int(parts[1])  # Extract the number after "demo_"
        
        # Increment the demonstration number
        new_demo_number = demo_number + 2041
        
        # Create the new file name
        new_filename = f"demo_{new_demo_number}_" + "_".join(parts[2:])
        
        # Rename the file
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")
