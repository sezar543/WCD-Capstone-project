import typing as t
from typing import Tuple, List
from pathlib import Path
import numpy as np

# import joblib
import pickle

import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.pipeline import Pipeline

from nn_model import __version__ as _version
from nn_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config
from nn_model.labels_utile import set_labels, get_labels

import h5py 
from tensorflow.python.keras.saving import hdf5_format

from tensorflow.python.keras.models import save_model, load_model

import os

import gc
import tensorflow as tf
from tensorflow import keras

# Import necessary libraries
import pickle
from nn_model.config.core import config
from nn_model.config.core import TRAINED_MODEL_DIR
from nn_model.processing.nn_architecture import nn_architecture  # Replace with the actual location of your Pipeline class
# Import base_dir from config.py
from nn_model.config.core import base_dir

gc.collect()  # Manually collect garbage

# # Define the base path for the dataset
# path = "C:\Course_online\Deploying-Machine-learning-models-thirdClone\nn_model\datasets"
# path1 = "C:\Course_online\Deploying-Machine-learning-models-thirdClone"
# path2 = "\nn_model\datasets"

# TRAINING_DIR = config.app_config.training_dir
# VERSION = config.app_config.version

def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.h5"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    save_model(pipeline_to_persist, str(save_path))

def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""
    file_path = TRAINED_MODEL_DIR / file_name
    print("load_pipeline .....file_path = ", file_path)
    trained_model = load_model(str(file_path))
    return trained_model


def load_single_image(image_path):
    """Load a single image using OpenCV."""
    try:
        img = cv2.imread(image_path)

        # Handle the case where the image is not loaded
        if img is None:
            print(f"Error reading image: {image_path}")
            return None

        # Resize or preprocess the image as needed
        # You might need to resize or preprocess based on your model requirements
        # Example: img = cv2.resize(img, (width, height))

        # Convert the BGR image to RGB
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize the pixel values if needed
        # Example: img_rgb = img_rgb / 255.0

        # Return the image array
        # return img_rgb
        return img
    except Exception as e:
        print(f"Error loading image from {image_path}: {str(e)}")
        return None



def get_tags(*, dataframe: pd.DataFrame):
    # Add a new column 'list_tags' to the DataFrame by splitting the 'tags' column on the space character
    dataframe["list_tags"] = dataframe.tags.str.split(" ")

    # Get the values of the new column
    row_tags = dataframe.list_tags.values

    # Flatten the list of tags
    tags = [tag for row in row_tags for tag in row]

    # Drop the created "list_tags" column
    dataframe = dataframe.drop("list_tags", axis='columns')
    
    return tags

def pre_pipeline_preparation(*, dataframe: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    # Initialization and Image Reading

    # Initialize empty lists to store the training images and their labels
    x_train = []
    y_train = []

    # Flatten the list of tags
    flatten = lambda l: [item for sublist in l for item in sublist]

    tags = get_tags(dataframe = dataframe)

    labels = list(set(flatten([l.split(' ') for l in dataframe['tags'].values])))
    print("labels are : ", labels)
    # Set the list of tags, i.e. labels, in utile.labels, to be used later as cachec_labels
    cached_labels = None  # Initialize with None
    
    print("Labels before set_labels:", labels)
    cached_labels = set_labels(labels)
    # print("Labels after set_labels:", cached_labels)

    # set_labels(labels)
    # Ensure that cached_labels is not None before proceeding
    
    if cached_labels is None:
        raise ValueError("set_labels returned None. Ensure it returns the updated labels.")

    cached_labels = get_labels()
    # print("test=", cached_labels == labels)

    # Create a label map for the unique tags in the dataset
    label_map = {l: i for i, l in enumerate(cached_labels)}
    # inv_label_map = {i: l for l, i in label_map.items()}
    length_label_map = len(label_map)

    # Loop through the training DataFrame
    for row in tqdm(dataframe.itertuples(), miniters=1000):
            f = row.Index  # Assuming the index is used as 'f'
            tags = row.tags  # Assuming the column name for tags is 'tags'
            
            # Read the image file
            try:
                # current_dir = os.path.dirname(os.path.abspath(__file__))

                # # Construct the relative path for the image
                # path_to_image = os.path.join(current_dir, 'nn_model', 'datasets', 'train-jpg', 'train_{}.jpg'.format(f))

                # Define a relative path to the train-jpg folder
                # base_dir = os.path.dirname(os.path.abspath(__file__))
                image_dir = os.path.join(base_dir, 'datasets', 'train-jpg')

                # Assuming 'f' is your image file identifier (integer or string)
                path_to_image = os.path.join(image_dir, f'train_{f}.jpg')

                # path_to_image=r'C:\Course_online\Deploying-Machine-learning-models-thirdClone\nn_model\datasets\train-jpg\train_{}.jpg'.format(f)

                img = cv2.imread(path_to_image)
                if img is None:
                    print(f"Error reading image: {path_to_image}")
            except Exception as e:
                print(f"Error: {e}")

            # img = cv2.imread(r'C:\Course_online\Deploying-Machine-learning-models-thirdClone\nn_model\datasets\train-jpg\train_{}.jpg'.format(f))

            # Initialize an array of zeros for the targets
            targets = np.zeros(length_label_map)
            # Loop through the tags for the current image
            for t in tags.split(' '):
                # Set the corresponding target value to 1
                targets[label_map[t]] = 1 
            # Append the image and its labels to the appropriate lists

            # [remove it] 
            # print("image shape is ", img.shape)
            x_train.append(cv2.resize(img, (64, 64)))  # Indicate the IMG Size
            y_train.append(targets)

    # Convert the lists to numpy arrays
    x_train = np.array(x_train, np.float16) / 255.
    y_train = np.array(y_train, np.uint8)

    return x_train , y_train, labels 


def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    # dataframe["MSSubClass"] = dataframe["MSSubClass"].astype("O")

    # rename variables beginning with numbers to avoid syntax errors later
    # transformed = dataframe.rename(columns=config.model_config.variables_to_rename)
    # return transformed
    return dataframe



def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()

# def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
#     """Persist the pipeline.
#     Saves the versioned model, and overwrites any previous
#     saved models. This ensures that when the package is
#     published, there is only one trained model that can be
#     called, and we know exactly how it was built.
#     """
#     # Prepare versioned save file name
#     save_file_name = f"{config.app_config.pipeline_save_file}{_version}.h5"
#     save_path = TRAINED_MODEL_DIR / save_file_name

#     remove_old_pipelines(files_to_keep=[save_file_name])

#     print("!!!save_path!!! = ", save_path)
#     # Assuming your pipeline has a model attribute
#     pipeline_to_persist.named_steps["nn_architecture"].save(save_path)

# def load_pipeline(*, file_name: str) -> Pipeline:
#     """Load a persisted pipeline."""

#     file_path = TRAINED_MODEL_DIR / file_name

#     # Check if the file exists
#     if not file_path.exists():
#         raise FileNotFoundError(f"Model file not found: {file_path}")

#     trained_model = tf.keras.models.load_model(file_path)

#     return trained_model

# # Function to save the pipeline
# def save_pipeline(*, pipeline_to_persist: nn_architecture) -> None:
#     """Persist the pipeline."""
#     # Prepare versioned save file name
#     save_file_name = f"{config.app_config.pipeline_save_file}{config.VERSION}.pkl"
#     save_path = TRAINED_MODEL_DIR / save_file_name

#     # Remove old pipelines
#     remove_old_pipelines(files_to_keep=[save_file_name])

#     # Save the pipeline using pickle
#     with open(save_path, 'wb') as file:
#         pickle.dump(pipeline_to_persist, file)

# # Function to load the pipeline
# def load_pipeline(*, file_name: str) -> nn_architecture:
#     """Load a persisted pipeline."""
#     file_path = TRAINED_MODEL_DIR / file_name

#     # Load the pipeline using pickle
#     with open(file_path, 'rb') as file:
#         trained_model = pickle.load(file)

#     return trained_model


# def load_pipeline(*, file_name: str) -> Pipeline:
#     """Load a persisted pipeline."""

#     file_path = TRAINED_MODEL_DIR / file_name
#     with open(file_path, 'rb') as file:
#         trained_model = pickle.load(file)
#     return trained_model

# def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
#     """Persist the pipeline.
#     Saves the versioned model, and overwrites any previous
#     saved models. This ensures that when the package is
#     published, there is only one trained model that can be
#     called, and we know exactly how it was built.
#     """

#     # Prepare versioned save file name
#     save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
#     save_path = TRAINED_MODEL_DIR / save_file_name

#     remove_old_pipelines(files_to_keep=[save_file_name])
#     with open(save_path, 'wb') as file:
#         pickle.dump(pipeline_to_persist, file)





# def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
#     """Persist the pipeline.
#     Saves the versioned model, and overwrites any previous
#     saved models. This ensures that when the package is
#     published, there is only one trained model that can be
#     called, and we know exactly how it was built.
#     """

#     # Prepare versioned save file name
#     save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
#     save_path = TRAINED_MODEL_DIR / save_file_name

#     remove_old_pipelines(files_to_keep=[save_file_name])
#     joblib.dump(pipeline_to_persist, save_path)


# def load_pipeline(*, file_name: str) -> Pipeline:
#     """Load a persisted pipeline."""

#     file_path = TRAINED_MODEL_DIR / file_name
#     trained_model = joblib.load(filename=file_path)
#     return trained_model