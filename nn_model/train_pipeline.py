import sys
import pandas as pd
import os

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
# import os
import matplotlib.pyplot as plt
import mlflow

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from sklearn.metrics import fbeta_score
from nn_model.processing.evaluation_metrics import fbeta, accuracy_score  # Import custom metrics

from nn_model.config.core import config
# from nn_model.pipeline import nn_pipe
from nn_model.processing.data_manager import load_dataset, save_pipeline
from sklearn.model_selection import train_test_split
from nn_model.processing.data_manager import pre_pipeline_preparation
from nn_model.processing.nn_architecture import nn_architecture
from nn_model.labels_utile import get_labels
from nn_model.config.core import PREDICTIONS_DIR
from deploy_mlflow.utils import get_mlflow_db_path

import gc

# Assuming you have a global variable that holds your model
global_model = None

import tensorflow as tf
import random 

import sys
print(f"Python version: {sys.version}")

# Set seeds
random_seed = 42  # You can use any integer value
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

#print the versions of tensorflow and numpy
# TensorFlow version: 2.10.0 NumPy version: 1.26.2
# import tensorflow as tf
# from tensorflow import keras

# print(f"TensorFlow version: {tf.__version__}")
# print(f"NumPy version: {np.__version__}")
# # print(f"Keras version: {keras.__version__}")
# print(f"Python executable: {sys.executable}")
# import os
# print(f"PATH: {os.environ['PATH']}")
# import sys
# print(f"sys.prefix: {sys.prefix}")
# print(f"Python version: {sys.version}")

# # Define the base path for the dataset

# path2 = "\deploying-machine-learning-models\section-05-production-model-package\nn_model\datasets"
# path2 = r"\nn_model\datasets"

# Set MLflow tracking URI
# mlflow_db_path = get_mlflow_db_path()
# mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")

# print("MLflow tracking URI set to:", mlflow_db_path)

# # Get the current directory of train_pipeline.py
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # Construct the relative path to mlflow.db
# relative_path = os.path.join(current_dir, '..', 'deploy_mlflow', 'mlruns', 'mlflow.db')
# print("relative_path insde train_pipeline = ", relative_path)
# # Set MLflow tracking URI with the relative path to the SQLite database
# mlflow.set_tracking_uri(f"sqlite:///{os.path.normpath(relative_path)}")

# mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))


def load_and_preprocess_data():
    print("config.app_config.training_data_file = ", config.app_config.training_data_file)
    df_train_sample = load_dataset(file_name=config.app_config.training_data_file)
    
    df_train = df_train_sample.head(800).copy()

    X, Y, labels = pre_pipeline_preparation(dataframe=df_train)

    x_train, x_val, y_train, y_val = train_test_split(
        X,
        Y,
        test_size=config.model_config.test_size,
        shuffle=True,
        random_state=config.model_config.random_state,
    )
    gc.collect()

    return x_train, y_train, x_val, y_val

def train_nn_model(x_train, y_train, x_val, y_val):
    nn_model_instance = nn_architecture()
    
    # Compile the model
    nn_model_instance.compile_model()

    # #For memory savings:
    # tf.keras.backend.clear_session()

    # Train the model
    history = nn_model_instance.fit(x_train, y_train, x_val, y_val)

    # Save the trained model
    nn_model_instance.save_model()

    # save_path = r"C:\Course_online\Deploying-Machine-learning-models-thirdClone\nn_model\trained-models\nn_model_output_v0.0.1.h5"
    # nn_model_instance.save(save_path)

    gc.collect()

    return nn_model_instance.model, history

def run_training(show_plot=True) -> None:
# def run_training() -> None:
    """Train the model."""

    # Read the train_classes.csv file and store it in a DataFrame
    # df_train = pd.read_csv(file_name=config.app_config.path_train_class)
    
    # df_train = load_dataset(file_name=config.app_config.training_data_file)
    
    x_train, y_train, x_val, y_val = load_and_preprocess_data()

    nn_model, history = train_nn_model(x_train, y_train, x_val, y_val)

    # nn_model_instance = nn_architecture()

    # # Compile the model
    # nn_model_instance.compile_model()

    # history = nn_model_instance.fit(x_train, y_train, x_val, y_val)

    # nn_model_instance.model.summary()
    # # print("Is this printed? ")

    # # Save the trained model
    # save_path = r"C:\Course_online\Deploying-Machine-learning-models-thirdClone\nn_model\trained-models\nn_model_output_v0.0.1.h5"
    # nn_model_instance.save(save_path)

    if show_plot:
        plt.plot(history.history['loss'])
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'])

        # Adding title, y-label and x-label to the plot
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')

        # Adding legend to the plot
        plt.legend(['train', 'validation'] if 'val_loss' in history.history else ['train'], loc='upper left')

        # Showing the plot
        plt.show()
    


    # Assuming model is your trained model and X_train is your training data
    predictions = np.round(nn_model.predict(x_train))
    print("length of predictions = ", len(predictions))
    print("First predictions = ", predictions[0])
    print("length x_train = ", len(x_train))
    print("length x_val= ", len(x_val))


    # # Load the corresponding image names from the x_train dataset
    # # Assuming x_train is a DataFrame with a column 'file_path' representing the file paths
    # image_names = [os.path.basename(file_path) for file_path in x_train['file_path']]

    df_train_sample = load_dataset(file_name=config.app_config.training_data_file)
    # print(df_train_sample["image_name"].head(5))

    df_train = df_train_sample.head(len(predictions)).copy()
    # X_train, _, _ = pre_pipeline_preparation(dataframe=df_train)

    # Load the original data to get the image names
    image_names = df_train.loc[:, 'image_name'].tolist()  
    print("length of image_names = " ,len(image_names))

    # Ensure the predictions folder exists
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    # Construct the full path for saving the CSV file inside the predictions folder
    csv_predictions_df_path = os.path.join(PREDICTIONS_DIR, "predictions.csv")
    
    # Assuming 'predictions' is a NumPy array or Pandas DataFrame with 17 columns
    cached_labels = get_labels()

    # Assuming 'predictions' is a NumPy array or Pandas DataFrame
    predictions_df = pd.DataFrame(predictions)
    # Add the image names as a new column in the predictions DataFrame
    predictions_df['image_name'] = image_names
    # Rearrange the columns to make 'image_name' the first column
    predictions_df = predictions_df[['image_name'] + [col for col in predictions_df.columns if col != 'image_name']]

    # Create a new column 'predicted_tags' with the tags where prediction is 1
    predictions_df['predicted_tags'] = predictions_df.apply(lambda row: ' '.join(tag for tag, pred in zip(cached_labels, row[:-2]) if pred == 1), axis=1)
    print(predictions_df.head())

    # Keep only 'image_name' and 'predicted_tags' columns
    predictions_df = predictions_df[['image_name', 'predicted_tags']]
    predictions_df.to_csv(csv_predictions_df_path, index=False)


    csv_predictions_dfT_path = os.path.join(PREDICTIONS_DIR, "predictionsTwo.csv")
    predictions_dfT = pd.DataFrame(predictions, columns = cached_labels)
    predictions_dfT['image_name'] = image_names
    # Rearrange the columns to make 'image_name' the first column
    predictions_dfT = predictions_dfT[['image_name'] + [col for col in predictions_dfT.columns if col != 'image_name']]
    predictions_dfT['predicted_tags'] = predictions_dfT.apply(lambda row: ' '.join(tag for tag, pred in zip(cached_labels, row[:-2]) if pred == 1), axis=1)

    predictions_dfT.to_csv(csv_predictions_dfT_path, index=False)

    # # Access the neural network model directly from the pipeline
    # nn_model = nn_model_instance.model

    # Calculate the f-beta score for the training set
    train_fbeta =  fbeta(y_train, predictions)
    print("train fbeta = ", train_fbeta)

    # Calculate the f-beta score for the validation set
    val_fbeta = fbeta(y_val, np.round(nn_model.predict(x_val)))
    print("val fbeta: ", val_fbeta)

    train_accuracy_score = accuracy_score(y_train, predictions, epsilon=1e-4)
    print("train_accuracy_score = ", np.mean(train_accuracy_score))

    val_accuracy_score = accuracy_score(y_val, np.round(nn_model.predict(x_val)), epsilon=1e-4)
    print("val_accuracy_score = ", np.mean(val_accuracy_score))


# if __name__ == "__main__":
#     run_training()

if __name__ == "__main__":
    # Check if the "--no-plot" command-line argument is provided
    show_plot = "--no-plot" not in sys.argv

    run_training(show_plot=show_plot)

#     1/6 [====>.........................] - ETA: 13s - loss: 0.7015 - fbeta: 0.2407 - accuracy_score: 0.4577
# 2/6 [=========>....................] - ETA: 2s - loss: 0.5351 - fbeta: 0.4376 - accuracy_score: 0.6797 
# 3/6 [==============>...............] - ETA: 1s - loss: 0.4783 - fbeta: 0.4961 - accuracy_score: 0.7552
# 4/6 [===================>..........] - ETA: 1s - loss: 0.4363 - fbeta: 0.5107 - accuracy_score: 0.7908
# 5/6 [========================>.....] - ETA: 0s - loss: 0.4137 - fbeta: 0.5262 - accuracy_score: 0.8113
# 6/6 [==============================] - ETA: 0s - loss: 0.4038 - fbeta: 0.5369 - accuracy_score: 0.8189
# 6/6 [==============================] - 6s 698ms/step - loss: 0.4038 - fbeta: 0.5369 - accuracy_score: 0.8189 - val_loss: 0.6051 - val_fbeta: 0.4725 - val_accuracy_score: 0.8529
# Model: "sequential_1"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  batch_normalization_1 (Bat  (None, 64, 64, 3)         12
#  chNormalization)

#  conv2d_4 (Conv2D)           (None, 64, 64, 32)        896

#  conv2d_5 (Conv2D)           (None, 62, 62, 32)        9248

#  max_pooling2d_2 (MaxPoolin  (None, 31, 31, 32)        0
#  g2D)

#  dropout_3 (Dropout)         (None, 31, 31, 32)        0

#  conv2d_6 (Conv2D)           (None, 31, 31, 64)        18496

#  conv2d_7 (Conv2D)           (None, 29, 29, 64)        36928

#  max_pooling2d_3 (MaxPoolin  (None, 14, 14, 64)        0
#  g2D)

#  dropout_4 (Dropout)         (None, 14, 14, 64)        0

#  flatten_1 (Flatten)         (None, 12544)             0

#  dense_2 (Dense)             (None, 512)               6423040

#  dropout_5 (Dropout)         (None, 512)               0

#  dense_3 (Dense)             (None, 17)                8721

# =================================================================
# Total params: 6497341 (24.79 MB)
# Trainable params: 6497335 (24.79 MB)
# Non-trainable params: 6 (24.00 Byte)
# _________________________________________________________________

#  1/23 [>.............................] - ETA: 4s
#  3/23 [==>...........................] - ETA: 0s
#  5/23 [=====>........................] - ETA: 0s
#  7/23 [========>.....................] - ETA: 0s
#  8/23 [=========>....................] - ETA: 0s
#  9/23 [==========>...................] - ETA: 0s
# 10/23 [============>.................] - ETA: 0s
# 12/23 [==============>...............] - ETA: 0s
# 13/23 [===============>..............] - ETA: 0s
# 14/23 [=================>............] - ETA: 0s
# 15/23 [==================>...........] - ETA: 0s
# 17/23 [=====================>........] - ETA: 0s
# 18/23 [======================>.......] - ETA: 0s
# 19/23 [=======================>......] - ETA: 0s
# 20/23 [=========================>....] - ETA: 0s
# 21/23 [==========================>...] - ETA: 0s
# 22/23 [===========================>..] - ETA: 0s
# 23/23 [==============================] - ETA: 0s
# 23/23 [==============================] - 1s 58ms/step
# accuracy mean:  0.4203051510720993

#  1/23 [>.............................] - ETA: 1s
#  3/23 [==>...........................] - ETA: 0s
#  5/23 [=====>........................] - ETA: 0s
#  7/23 [========>.....................] - ETA: 0s
#  9/23 [==========>...................] - ETA: 0s
# 11/23 [=============>................] - ETA: 0s
# 13/23 [===============>..............] - ETA: 0s
# 15/23 [==================>...........] - ETA: 0s
# 17/23 [=====================>........] - ETA: 0s
# 19/23 [=======================>......] - ETA: 0s
# 21/23 [==========================>...] - ETA: 0s
# 23/23 [==============================] - ETA: 0s
# 23/23 [==============================] - 1s 37ms/step
# train fscore:  0.4203051510720993

# 1/3 [=========>....................] - ETA: 0s
# 3/3 [==============================] - ETA: 0s
# 3/3 [==============================] - 0s 39ms/step
# val fscore:  0.41181551009004747