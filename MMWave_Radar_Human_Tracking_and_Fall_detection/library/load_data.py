import pickle
import os
import pandas as pd

# # Parameters used in saving the file
# experiment_name = "test"  # Replace with your actual experiment name
# file_label = "auto_RadarSeq"
# file_time = "Jun-20-00-50-36"  # Replace with the actual time string when the file was saved
# file_dir = "data/Test/2024_Jun/"  # Replace with the actual directory path

# # Construct the filename
# file_name = f"{experiment_name}_{file_label}_{file_time}"
# file_path = os.path.join(file_dir, file_name)
# file_path = "/Users/katiezhou/P4P-mmWave/MMWave_Radar_Human_Tracking_and_Fall_detection/pointnet_data/sit/train/standing_auto_RadarSeq_Jul-24-00-50-36_0"
# file_path = "/Users/katiezhou/P4P-mmWave/MMWave_Radar_Human_Tracking_and_Fall_detection/data/Maggs_307/2024_Jul/test_manual_RadarSnapshot_Jul-23-17-03-12"
file_path = "/Users/katiezhou/P4P-mmWave/MMWave_Radar_Human_Tracking_and_Fall_detection/pointnet_data/stand/test/standing_auto_RadarSeq_Jul-24-00-50-36_2"
# Load the data from the pickle file
with open(file_path, 'rb') as file:
    data = pickle.load(file)
    df = pd.DataFrame(data)
    # sampled_data_df = df.sample(n=40, random_state=1)
    # sampled_data = df.sample(20)
    # data = data.iloc[0]

# Now 'data' contains the list of radar data frames that were saved
# print(sampled_data_df = df.sample(n=40, random_state=1))
print(df)
# print(sampled_data)
