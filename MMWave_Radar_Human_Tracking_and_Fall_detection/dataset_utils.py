from glob import glob

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import pickle



def tf_parse_filename(filename):
    """Take batch of filenames and create point cloud and label"""

    idx_lookup = {'standing': 0, 'sitting': 1, 'laying down': 2}

    def parse_filename(filename_batch):

        pt_clouds = []
        labels = []
        for filename in filename_batch:
            # Read in point cloud
            filename_str = filename.numpy().decode()
            pt_cloud = np.load(filename_str)

            # Add rotation and jitter to point cloud
            theta = np.random.random() * 2*3.141
            A = np.array([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta), np.cos(theta), 0],
                          [0, 0, 1]])
            offsets = np.random.normal(0, 0.02, size=pt_cloud.shape)
            pt_cloud = np.matmul(pt_cloud, A) + offsets

            # Create classification label
            obj_type = filename_str.split('/')[1]   # e.g., airplane, bathtub
            label = np.zeros(40, dtype=np.float32)
            label[idx_lookup[obj_type]] = 1.0

            pt_clouds.append(pt_cloud)
            labels.append(label)

        return np.stack(pt_clouds), np.stack(labels)


    x, y = tf.py_function(parse_filename, [filename], [tf.float32, tf.float32])
    return x, y


def tf_parse_filename_test(filename):
    """Take batch of filenames and create point cloud and label"""

    idx_lookup = {'standing': 0, 'sitting': 1, 'laying down': 2}

    def parse_filename(filename_batch):

        pt_clouds = []
        labels = []
        for filename in filename_batch:
            # Read in point cloud
            filename_str = filename.numpy().decode()
            pt_cloud = np.load(filename_str)

            # Create classification label
            obj_type = filename_str.split('/')[1]   # e.g., airplane, bathtub
            label = np.zeros(40, dtype=np.float32)
            label[idx_lookup[obj_type]] = 1.0

            pt_clouds.append(pt_cloud)
            labels.append(label)

        return np.stack(pt_clouds), np.stack(labels)


    x, y = tf.py_function(parse_filename, [filename], [tf.float32, tf.float32])
    return x, y


def train_val_split(train_size=0.92):
    train, val = [], []
    for obj_type in glob('ModelNet40/*/'):
        cur_files = glob(obj_type + 'train/*.npy')
        cur_train, cur_val = \
            train_test_split(cur_files, train_size=train_size, random_state=0, shuffle=True)
        train.extend(cur_train)
        val.extend(cur_val)

    return train, val

def df_combined(data):
    df = pd.DataFrame(data)
    ori = df['IWR1843_Ori'].tolist()
    side = df['IWR1843_Side'].tolist()
    combined = ori + side
    result = pd.DataFrame({'IWR1843_Ori': combined})

    return result

def parse_dataset(num_points, DATA_DIR):
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    folders = glob(os.path.join(DATA_DIR, "[!README]*"))

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("/")[-1]
        # gather all files
        train_files = glob(os.path.join(folder, "train/*"))
        test_files = glob(os.path.join(folder, "test/*"))

        for f in train_files:
            with open(f, 'rb') as file:
                data = pickle.load(file)
                nested_array = data.values[0].tolist()
                nested_array_np = np.array(nested_array, dtype=np.float16)
                sampled_indices = np.random.choice(nested_array_np.shape[0], size=num_points, replace=False)
                sampled_data = nested_array_np[sampled_indices]
                # print(data)
                # print("loaded: {}".format(f))
                # print(num_points)
                # sampled_data = data.sample(num_points).to_numpy()
                train_points.append(sampled_data)
                train_labels.append(i)

        for f in test_files:
            with open(f, 'rb') as file:
                data = pickle.load(file)
                nested_array = data.values[0].tolist()
                nested_array_np = np.array(nested_array, dtype=np.float16)
                sampled_indices = np.random.choice(nested_array_np.shape[0], size=num_points, replace=False)
                sampled_data = nested_array_np[sampled_indices]
                # print(data)
                # print("loaded: {}".format(f))
                # print(num_points)
                # sampled_data = data.sample(num_points).to_numpy()
                test_points.append(sampled_data)
                test_labels.append(i)
                # print(test_labels)

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )

def voxelize(point_cloud, voxel_size):
    """
    Voxelize the given point cloud data.

    :param point_cloud: (ndarray) An array of shape (n, 3) where each row represents a point (x, y, z).
    :param voxel_size: (float) The size of each voxel.
    :return: (ndarray) A voxel grid where each cell represents a voxel.
    """
    # Determine the range of the data
    min_coords = np.min(point_cloud, axis=0)
    max_coords = np.max(point_cloud, axis=0)
    
    # Calculate the number of voxels needed in each dimension
    voxel_shape = np.ceil((max_coords - min_coords) / voxel_size).astype(int)
    
    # Initialize voxel grid
    voxel_grid = np.zeros(voxel_shape, dtype=np.float32)
    
    # Compute voxel indices
    voxel_indices = np.floor((point_cloud - min_coords) / voxel_size).astype(int)
    
    # Populate voxel grid
    for idx in range(point_cloud.shape[0]):
        x, y, z = voxel_indices[idx]
        voxel_grid[x, y, z] += 1  # Increment voxel count
    
    return voxel_grid

def visualize_voxel_grid(voxel_grid, voxel_size):
    """
    Visualize a voxel grid using Open3D.

    :param voxel_grid: (ndarray) A voxel grid.
    :param voxel_size: (float) The size of each voxel.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Create voxel grid
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(np.argwhere(voxel_grid > 0) * voxel_size))
    )
    
    vis.add_geometry(voxel_grid)
    vis.run()
    vis.destroy_window()