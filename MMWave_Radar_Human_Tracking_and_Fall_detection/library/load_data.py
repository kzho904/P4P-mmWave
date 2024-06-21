import pickle
import os

# Parameters used in saving the file
experiment_name = "test"  # Replace with your actual experiment name
file_label = "auto_RadarSeq"
file_time = "Jun-20-00-50-36"  # Replace with the actual time string when the file was saved
file_dir = "data/Test/2024_Jun/"  # Replace with the actual directory path

# Construct the filename
file_name = f"{experiment_name}_{file_label}_{file_time}"
file_path = os.path.join(file_dir, file_name)

# Load the data from the pickle file
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Now 'data' contains the list of radar data frames that were saved
print(data)