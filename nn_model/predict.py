import os
import sys

# Print the current working directory
print("Current working directory:", os.getcwd())

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Check if the path is added
print("System path:", sys.path)

import typing as t
import gc

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from nn_model.pipeline import nn_pipe
from nn_model import __version__ as _version
from nn_model.config.core import config
from nn_model.processing.nn_architecture import load_SavedModel

from nn_model.processing.data_manager import pre_pipeline_preparation
from nn_model.processing.data_manager import load_single_image
from nn_model.labels_utile import get_labels
from nn_model.processing.evaluation_metrics import fbeta, accuracy_score
from nn_model.processing.nn_architecture import nn_architecture
from nn_model.train_pipeline import train_nn_model, load_and_preprocess_data

from nn_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config, base_dir

gc.collect()  # Manually collect garbage

# pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.h5"
print("!!!!!!!!pipeline_file_name!!!!!!! = ", pipeline_file_name)
# _nn_pipe = load_pipeline(file_name=pipeline_file_name)

# Assuming you have a global variable that holds your model
global_model = None

def load_or_create_model(saved_model=True):
    print("!!!Inside load_or_create_model!!!")

    global global_model
    if global_model is None:
        if not saved_model:
            print("Hey! Did not use the saved method!")
            global_model, _ = train_nn_model(*load_and_preprocess_data())
        else:
            print("Hey! used the saved method!")
            global_model = load_trained_model(pipeline_file_name)

    return global_model

def load_trained_model(pipeline_file_name):
    """Load the trained model pipeline."""
    print("Inside load_trained_model")
    model = load_SavedModel(file_name=pipeline_file_name)
    print("Loaded model:", model)

    return model

# def preprocess_image(image_path):
#     """Preprocess a single image."""
#     single_image = load_single_image(image_path)  # Replace with your actual image loading function
#     preprocessed_image = pre_pipeline_preparation(dataframe=single_image)
#     return preprocessed_image

def make_prediction(img, saved_model, threshold=0.5):
    # Load the data_manager module to access the necessary functions
    # from nn_model.processing.data_manager import load_single_image, get_labels
    print("!!!Inside make_prediction!!!")
    # Load or create the model
    model = load_or_create_model(saved_model = saved_model)
    # print("model =" , model)
    # Load the labels
    labels = get_labels()

    # Load a single image using the provided function
    # img = load_single_image(image_path)

    # Ensure the image is in RGB mode if it's not
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Convert the PIL image to a NumPy array
    img_np = np.array(img)

    img_resized = cv2.resize(img_np, (64, 64))  # Indicate the IMG Size
    img_resized = np.array(img_resized, np.float16) / 255.

    img_resized_batch = np.expand_dims(img_resized, axis=0)

    # Make predictions for the preprocessed image
    predictions = model.predict(img_resized_batch)

    # Post-process predictions
    binary_predictions = (predictions > threshold).astype(int)

    # Get the predicted tags
    predicted_tags = [tag for i, tag in enumerate(labels) if binary_predictions[0, i] == 1]

    print("predictions = ", predictions)
    # print("binary predictions = ", binary_predictions)
    print("predicted_tags = ", predicted_tags)

    return predicted_tags, predictions

if __name__ == "__main__":
    # Example usage

    for f in range(8):
        # image_path = os.path.join('nn_model', 'datasets', 'train-jpg', f'train_{f}.jpg')
        # image_path=r'C:\Course_online\Deploying-Machine-learning-models-thirdClone\nn_model\datasets\train-jpg\train_{}.jpg'.format(f)
        current_dir = os.path.dirname(os.path.abspath(__file__))

        print("PACKAGE_ROOT3 = ", base_dir)

        # Construct the relative path for the image
        image_path = os.path.join(current_dir, 'datasets', 'train-jpg', 'train_{}.jpg'.format(f))


        # print("Print 1")
        if os.path.exists(image_path):
            # print("Print 2")
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error reading image: {image_path}")
        else:
            print(f"Error: Image not found at {image_path}")

        # image_path=r'C:\Course_online\Deploying-Machine-learning-models-thirdClone\nn_model\datasets\train-jpg\train_0.jpg'

        print("Printed here?")

        # f = 10
        print("f =", f)
        prediction = make_prediction(image_path, pipeline_file_name)
        
        # print("Predicted Tags for image number ", f, " are :", prediction)

# def load_or_create_model(saved_model = True):
#     global global_model
#     if global_model is None:
#         if not saved_model:
#             print("Hey! Did not use the saved method!")

#             # path_model = r"C:\Course_online\Deploying-Machine-learning-models-thirdClone\nn_model\trained-models\nn_model_output_v0.0.1.h5"
#             global_model, _ = train_nn_model(*load_and_preprocess_data())
#         else:
#             print("Hey! used the saved method!")
#             # nn_model_instance = nn_architecture()
#             # nn_model_instance.load_SavedModel(file_name=pipeline_file_name)
#             # global_model = nn_model_instance.model
#             global_model = load_trained_model(pipeline_file_name)

#             # global_model = nn_architecture()
#             # global_model.load_SavedModel(file_name=pipeline_file_name)
#     return global_model

# def make_prediction(image_path, threshold=0.5):
#     """Make predictions for a single image."""
#     # Load the trained model
#     print("I got here!")

#     # nn_model = load_trained_model(pipeline_file_name)
#     file_path = TRAINED_MODEL_DIR / pipeline_file_name

#     nn_model = tf.keras.models.load_model(
#         file_path,
#         custom_objects={"fbeta": fbeta, "accuracy_score": accuracy_score},
#     )

#     # nn_model = tf.keras.models.load_model(
#     #     model_path,
#     #     custom_objects={"fbeta": fbeta, "accuracy_score": accuracy_score},
#     # )
#     # print(nn_model.summary())
#     # print(nn_model.summary())

#     # Preprocess the image

    
#     img = cv2.imread(str(image_path))
#     img_resized = cv2.resize(img, (64, 64))  # Indicate the IMG Size
#     # Convert the lists to numpy arrays
#     img_resized = np.array(img_resized, np.float16) / 255.
#     # preprocessed_image = preprocess_image(image_path)

#     img_resized_batch = np.expand_dims(img_resized, axis=0)
#     # Make predictions for the preprocessed image
#     # predictions = np.round(nn_model.predict(img_resized_batch))
#     predictions = nn_model.predict(img_resized_batch)
#     # Post-process predictions
#     binary_predictions = (predictions > threshold).astype(int)
    
#     print("predictions = ", predictions)

#     # # Post-process predictions
#     # binary_predictions = (predictions > threshold).astype(int)

#     # Assuming all_tags is the list of all 17 tags
#     labels = get_labels()

#     predicted_tags = [tag for i, tag in enumerate(labels) if binary_predictions[0, i] == 1]

#     print("predictions = ", predictions)
#     print("predicted_tags = ", predicted_tags)

#     return binary_predictions
#     # predicted_tags = [tag for i, tag in enumerate(labels) if predictions[0, i] == 1]

#     # print("predicted_tags = ", predicted_tags)

#     return predictions




# def make_prediction(
#     *,
#     input_data: t.Union[pd.DataFrame, dict],
# ) -> dict:
#     """Make a prediction using a saved model pipeline."""

#     data = pd.DataFrame(input_data)
#     validated_data, errors = validate_inputs(input_data=data)
#     results = {"predictions": None, "version": _version, "errors": errors}

#     if not errors:
#         predictions = _nn_pipe.predict(
#             X=validated_data[config.model_config.features]
#         )
#         results = {
#             "predictions": [np.exp(pred) for pred in predictions],  # type: ignore
#             "version": _version,
#             "errors": errors,
#         }

#     return results

# def make_prediction(
#     *,
#     input_data: t.Union[pd.DataFrame, dict],
# ) -> t.List[str]:
#     """Make a prediction using a saved model pipeline."""
#     image_path = path_to_image=r'C:\Course_online\Deploying-Machine-learning-models-thirdClone\nn_model\datasets\train-jpg\train_{}.jpg'.format(image_index)
#     img = cv2.imread(image_path)

#     img_resized = cv2.resize(img, (64, 64))  # Indicate the IMG Size

#     # Convert the lists to numpy arrays
#     x_train = np.array(img_resized, np.float16) / 255.

#     # data = pd.DataFrame(input_data)
#     result = np.round(nn_pipe.predict(x_train))
                      
#     # No need for validation in the prediction script
#     # predictions = _nn_pipe.predict(X=data[config.model_config.features])
#     # results = {
#     #     "predictions": [np.exp(pred) for pred in predictions],
#     #     "version": _version,
#     #     "errors": None,
#     # }
#     return result