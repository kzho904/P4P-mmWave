import os
import glob
import trimesh
import numpy as np
import tensorflow as tf

# Define the data directory
DATA_DIR = '/raw_3_class_data'

def parse_dataset(num_points=2048, frames_per_file=10):
    np.random.seed(42)
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    folders = glob.glob(os.path.join(DATA_DIR, "[!README]*"))

    for i, folder in enumerate(folders):
        print("Processing class: {}".format(os.path.basename(folder)))
        class_map[i] = folder.split("/")[-1]
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))

        for f in train_files:
            mesh = trimesh.load(f)
            frames = extract_frames_from_mesh(mesh, frames_per_file, num_points)
            for frame in frames:
                train_points.append(frame)
                train_labels.append(i)

        for f in test_files:
            mesh = trimesh.load(f)
            frames = extract_frames_from_mesh(mesh, frames_per_file, num_points)
            for frame in frames:
                test_points.append(frame)
                test_labels.append(i)

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )

def extract_frames_from_mesh(mesh, frames_per_file, num_points):
    # Example function to extract frames from a mesh
    # Adjust based on your data format and how frames are stored
    frames = []
    for _ in range(frames_per_file):
        frame_points = mesh.sample(num_points)
        frames.append(frame_points)
    return frames

# Number of points, frames per file, and classes
NUM_POINTS = 2048
NUM_CLASSES = 10
BATCH_SIZE = 32
FRAMES_PER_FILE = 10

# Parse dataset
train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(NUM_POINTS, FRAMES_PER_FILE)

# Convert to TensorFlow tensors
train_points = tf.convert_to_tensor(train_points, dtype=tf.float32)
train_labels = tf.convert_to_tensor(train_labels, dtype=tf.int32)
test_points = tf.convert_to_tensor(test_points, dtype=tf.float32)
test_labels = tf.convert_to_tensor(test_labels, dtype=tf.int32)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

# Shuffle and batch the datasets
train_dataset = train_dataset.shuffle(len(train_labels)).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# Verify the dataset structure
for points, labels in train_dataset.take(1):
    print(points.shape, labels.shape)
