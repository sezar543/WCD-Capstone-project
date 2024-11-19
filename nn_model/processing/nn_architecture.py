import numpy as np
import os
import pandas as pd
from typing import List
import typing as t

from nn_model import __version__ as _version

from nn_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

import gc
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalMaxPooling2D, BatchNormalization
from keras.optimizers import Adam, Adagrad, RMSprop
from nn_model.config.core import config

import mlflow
import mlflow.keras

import sys
# sys.path.append("C:\Course_online\Deploying-Machine-learning-models-thirdClone\nn_model\processing\evaluation_metrics.py")
from .evaluation_metrics import fbeta, accuracy_score
from deploy_mlflow.utils import get_mlflow_db_path

# os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'deploy_mlflow')))
# Change the working directory to deploy_mlflow
deploy_mlflow_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'deploy_mlflow'))
print("Changing working directory to:", deploy_mlflow_path)
os.chdir(deploy_mlflow_path)

# mlflow.end_run()
mlflow_db_path = get_mlflow_db_path()
mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")

# if run locally:
# mlflow.set_tracking_uri("http://localhost:5000")

print("MLflow tracking URI set to:", mlflow.get_tracking_uri())
print("Current working directory:", os.getcwd())
# print("MLflow tracking URI set to:", mlflow_db_path)

gc.collect()  # Manually collect garbage

# Limit GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print("Here! Checking gpus!!!!")
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def load_SavedModel(file_name: str):
    """Load a persisted pipeline."""
    
    file_path = TRAINED_MODEL_DIR / file_name
    print("file_path =", file_path)
    # Check if the file exists
    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found: {file_path}")
    
    print("Loading model from:", file_name)
    
    nn_model_instance = tf.keras.models.load_model(
        file_path,
        custom_objects={"fbeta": fbeta, "accuracy_score": accuracy_score},
    )
    print("Loaded model summary:", nn_model_instance.summary())

    # self.model = nn_model_instance

    return nn_model_instance
    
class nn_architecture():
    def __init__(self, variables: List[str] = [], reference_variable: str = None):
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.reference_variable = reference_variable
        print("instantiating the nn_archi")
        # Define The Model
        self.model = self.build_model()

    def build_model(self):
        # Define The Model
        print("model of nn_archi")
        model = tf.keras.Sequential()

        # Adding The Layers
        # Batch Normalization layer is added as the first layer of the model, which normalize the input data.
        model.add(BatchNormalization(input_shape=(64, 64, 3)))

        # Convolutional layers and MaxPooling layers are added to extract features from the input images and reduce the spatial dimensions of the feature maps respectively.
        model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Dropout layers are added to prevent overfitting.
        model.add(Dropout(0.2))

        # Same set of layers are added for the next set of features
        model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        # Flatten layer is added to convert the 2D feature maps into a 1D feature vector
        model.add(Flatten())

        # Fully connected layers (dense layers) and dropout layers are added
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(17, activation='sigmoid'))

        return model

    def compile_model(self):
        
        # Compiling the model
        self.model.compile(
            optimizer=Adam(),  # Use your desired optimizer
            loss='binary_crossentropy',
            metrics=[fbeta, accuracy_score]
        )

        # self.model = model
        
    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame = None, y_val: pd.Series = None):
        # Limit GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print("Here n2! Checking gpus!!!!")
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        tf.keras.backend.clear_session()

        self.compile_model()

        # MLflow Experiment tracking
        mlflow.set_experiment("new_experiment_name")  # Set your MLflow experiment name

        experiment = mlflow.get_experiment_by_name("new_experiment_name")

        # If the experiment already exists, use it; otherwise, create it with the correct artifact location
        if not experiment:
            mlflow.create_experiment(
                "new_experiment_name",
                artifact_location=f"file:///{mlflow_db_path}"  # Enforce artifact location
            )
            
        with mlflow.start_run() as run:
            # Log hyperparameters, if any
            mlflow.log_param("batch_size", 32)
            mlflow.log_param("epochs", 3)
            # Save a training log to log as an artifact
            log_file = "training_log.txt"
            with open(log_file, "w") as f:
                f.write("Training log information here.")

            history = self.model.fit(x_train, y_train,
                                    batch_size=32,
                                    epochs=3,
                                    verbose=1,
                                    validation_data=(x_val, y_val) if x_val is not None and y_val is not None else None)


            # Log the training log file as an artifact
            mlflow.log_artifact(log_file)
            
            # Attempt to access 'accuracy' or 'metrics'
            metrics = history.history  # Extracting the metrics
            # Print the list of available metrics
            print(f"Available metrics: {list(metrics.keys())}")

            # Number of epochs
            num_epochs = len(history.history['accuracy_score'])  # assuming 'loss' has one value per epoch

            # Log metrics for each epoch
            for epoch in range(num_epochs):
                # Debug: print metrics for each epoch
                print(f"History object after epoch {epoch + 1}: {history.history}")
                print(f"Type of history.history: {type(history.history)}")

                mlflow.log_metric("train_accuracy", history.history['accuracy_score'][epoch], step=epoch)
                mlflow.log_metric("val_accuracy", history.history['val_accuracy_score'][epoch], step=epoch)
                mlflow.log_metric("train_fbeta", history.history['fbeta'][epoch], step=epoch)
                mlflow.log_metric("val_fbeta", history.history['val_fbeta'][epoch], step=epoch)
                mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
                mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)

            # Create a dummy image as the input example
            dummy_image = np.random.rand(1, 64, 64, 3)  # shape (1, 64, 64, 3)
            # mlflow.keras.log_model(self.model, "my_model")

            # Register the model with a specific name
            model_uri = f"runs:/{run.info.run_id}/model"
            try:
                mlflow.keras.log_model(self.model, artifact_path="model", input_example=dummy_image)

                print("Model logging successful")
            except Exception as e:
                print(f"Model logging failed: {e}")

            mlflow.register_model(model_uri=model_uri, name="Amazon_image_classification")

            print(f"Run ID: {run.info.run_id}")

            # Clean up the log file
            os.remove(log_file)
                                           
        return history

    def save_model(self):

        # Prepare versioned save file name
        save_file_name = f"{config.app_config.pipeline_save_file}{_version}.h5"
        save_path = TRAINED_MODEL_DIR / save_file_name

        self.remove_old_pipelines(files_to_keep=[save_file_name])

        """Save the neural network model to a file."""
        self.model.save(save_path)
        print("!!!save_path!!! = ", save_path)


    def remove_old_pipelines(self, files_to_keep: t.List[str]) -> None:
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

        # self.model.fit(x_train, y_train,
        #                     batch_size=128,
        #                     epochs=10,
        #                     verbose=1,
        #                     validation_data=(x_val, y_val))
    
        # self.model.fit(x_train, y_train,
        #             batch_size=128,
        #             epochs=10,
        #             verbose=1)

        # return self

    # def transform(self, X: pd.DataFrame) -> np.ndarray:
    #     # Assuming X is your image data
    #     # You need to preprocess your image data before making predictions
    #     # For example, you may need to scale pixel values to the range [0, 1]

    #     # Transform the data using the neural network model
    #     return self.model.predict(X)


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
