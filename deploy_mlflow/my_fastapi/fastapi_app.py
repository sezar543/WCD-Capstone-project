# python -m uvicorn main:app --reload --host 0.0.0.0 --port 80

import os
import sys
# Server
from fastapi import FastAPI, HTTPException
# from mangum import Mangum
from loguru import logger
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
from contextlib import asynccontextmanager
import base64

from pydantic import BaseModel
from fastapi import File, UploadFile
import tensorflow as tf
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nn_model.config.core import config
# from nn_model.predict import make_prediction
from nn_model.predict import get_labels
from fastapi.responses import JSONResponse

# from keras import models
# from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import Model

# curl -X POST "http://127.0.0.1:8000/upload" -F "C:\Dell15\p\d-third\nn_model\datasets\train-jpg\train_0.jpg"


# from tensorflow.python.keras.models import load_model
from nn_model.processing.data_manager import load_pipeline
from nn_model.processing.evaluation_metrics import fbeta, accuracy_score
# import joblib
import pickle
# Import base_dir from config.py
from nn_model.config.core import base_dir
from nn_model import __version__ as _version
from fastapi.responses import FileResponse
import logging
import mlflow
import mlflow.pyfunc
from deploy_mlflow.utils import get_mlflow_db_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

# Set MLflow tracking URI
mlflow_db_path = get_mlflow_db_path()
# mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
# mlflow.set_tracking_uri("http://localhost:5000")

# mlflow.set_tracking_uri("http://mlflow:5000")

print("Fastapi: MLflow tracking URI set to:", mlflow_db_path)

# # Get the current directory of fastapi_app.py
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # Construct the path to mlflow.db relative to fastapi_app.py
# relative_path = os.path.join(current_dir, '..', 'mlruns', 'mlflow.db')

# # Set MLflow tracking URI with the relative path to the SQLite database
# mlflow.set_tracking_uri(f"sqlite:///{os.path.normpath(relative_path)}")

reconstructed_model = None  # Define globally so it’s accessible in predict_image
# Load the model by specifying the registered model name and version
model_name = "Amazon_image_classification"
model_version = 1
run_id = "9e563c9bcb8d45c0a2ecc4db13566e5a"

app = FastAPI()

@app.on_event("startup")
async def load_model():
    global reconstructed_model
    print("Inside startup event!")
    try:
        # Try loading from the remote MLflow server
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        reconstructed_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")
        print("Model loaded successfully from MLflow server!")
    except Exception as e:
        print(f"Error loading model from MLflow server: {e}")
        try:
            # Fallback to local SQLite database if the server loading fails
            mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
            
            # Check if model URI needs to be adjusted for Docker
            model_uri = f"file:///app/deploy_mlflow/mlruns/1/{run_id}/artifacts/model"  # Adjust this as needed
            model_uri = f"file:///app/deploy_mlflow/mlruns/1/{run_id}/artifacts/model"  # Corrected path for Docker

            print("model_uri (before fix) = ", model_uri)
            
            # If model URI points to a local file path (Windows-style), convert it to the container path
            if model_uri.startswith("file:///C:/"):
                model_uri = model_uri.replace("file:///C:/Dell15/p/IMAGE-CLASSIFICATION-DEPLOY", "/app")
                print("Adjusted model_uri = ", model_uri)

            # Load model from local SQLite
            reconstructed_model = mlflow.pyfunc.load_model(model_uri)
            print("Model loaded successfully with corrected URI!")

        except Exception as e:
            print(f"Error loading model from local SQLite database: {e}")
            reconstructed_model = None

# @app.on_event("startup")
# async def load_model():
#     print("Inside startup event!")
#     global reconstructed_model
#     # model_uri = "models:/Amazon_image_classification/1"  # Replace with actual model name and version
#     try:
#         # Load the model using MLflow
#         reconstructed_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")
#         print("Model loaded successfully!")
#     except Exception as e:
#         print(f"Error loading model: {e}")

# path_pkl = r'C:\Course_online\Deploying-Machine-learning-models-thirdClone\nn_model\trained-models\nn_model_output_v0.0.1.pkl'

# Initialize files
# clf = pickle.load(open(path_pkl, 'rb')) # model object

# model = joblib.load(path_pkl)

# Define the directory where uploaded files will be stored
# UPLOAD_DIRECTORY = "C:/Dell15/p/d-third/nn_model/datasets/train-jpg/"

# Get the current directory (where the script is running from)
current_dir = os.path.dirname(os.path.abspath(__file__))
print("current_dir = ", current_dir)
pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.h5"

# Construct the relative path to the .h5 file
relative_path_to_model = os.path.join('..', 'nn_model', 'trained-models', pipeline_file_name)
path_h5 = os.path.join(current_dir, relative_path_to_model)

# path_h5 = r'C:\Course_online\Deploying-Machine-learning-models-thirdClone\nn_model\trained-models\nn_model_output_v0.0.1.h5'


# Define the expected input format
class ImageDataModel(BaseModel):
    ImageData: str  # base64-encoded image data

# Define input model
class TextData(BaseModel):
    input: str

@app.post("/test")
async def reverse_text(input: TextData):
    print("text received is = ", input.input)
    reversed_text = input.input[::-1]  # Reverse the input text
    return {"reversed_text": reversed_text}


@app.post("/pred")
async def predict_image(image_data_model: ImageDataModel):
    print("Inside FastAPI!!!")
    global reconstructed_model

    if reconstructed_model is None:
        return {"error": "Model not loaded!"}

    try:
        # Extract base64-encoded image
        image_data_base64 = image_data_model.ImageData
        if not image_data_base64:
            return {"error": "ImageData not provided"}

        # Decode the base64 image data
        image_data = base64.b64decode(image_data_base64)

        # Open the image from the decoded bytes
        image = Image.open(BytesIO(image_data))

        # Convert the image to OpenCV format (BGR)
        # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Make predictions using the saved model
        tags, predictions = make_prediction(image)
        # print(f"FastAPI, predictions are {predictions}")

        if predictions is None:
            return {"error": "Failed to generate predictions"}
        
        # Return predictions and tags as JSON
        return {
            "predictions": predictions.tolist(),  # Convert to list if predictions are in NumPy format
            "tags": tags
        }
    
    except Exception as e:
        return {"error": str(e)}
       


@app.get('/')
def index():
    return {"message": "Welcome to this image classification project!"}


def make_prediction(img, threshold=0.5):
    # Load the data_manager module to access the necessary functions
    # from nn_model.processing.data_manager import load_single_image, get_labels
    print("!!!Inside fastapi/make_prediction!!!")
    print("Using model:", reconstructed_model)  # Add this to log the model being used

    # Load the labels
    labels = get_labels()
    print("labels are: ", labels)

    # Load a single image using the provided function
    # img = load_single_image(image_path)

    # Ensure the image is in RGB mode if it's not
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Convert the PIL image to a NumPy array
    img_np = np.array(img)

    img_resized = cv2.resize(img_np, (64, 64))  # Indicate the IMG Size
    img_resized = np.array(img_resized, np.float16) / 255.

    img_resized_batch = np.expand_dims(img_resized, axis=0).astype(np.float64)

    # print("Shape of input image batch:", img_resized_batch.shape)

    # print("Preparing to make predictions...")
    try:
        predictions = reconstructed_model.predict(img_resized_batch)
        # print("Predictions generated:", predictions)
    except Exception as e:
        print(f"Error during prediction: {e}")
        predictions = None

    # Post-process predictions
    binary_predictions = (predictions > threshold).astype(int)

    # Get the predicted tags
    predicted_tags = [tag for i, tag in enumerate(labels) if binary_predictions[0, i] == 1]

    print("predictions = ", predictions)
    # print("binary predictions = ", binary_predictions)
    print("predicted_tags = ", predicted_tags)

    return predicted_tags, predictions


# # Define the lifespan for the FastAPI app
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     print("Inside lifespan!!")
#     global reconstructed_model
#     model_uri = "models:/Amazon_image_classification /1"  # Replace with actual model name and version

#     try:
#         # Load the model using MLflow
#         reconstructed_model = mlflow.pyfunc.load_model(model_uri)
#         print("Model loaded successfully!")
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         raise HTTPException(status_code=500, detail="Model loading failed")

#     yield  # The app will run between these two points

#     print("Shutting down FastAPI app...")

# # Apply lifespan to FastAPI app
# app = FastAPI(lifespan=lifespan)




# @app.post("/pred")
# async def predict_image(input: InputModel):
#     # print("input = ", input)
#     try:
#         # Decode the base64 image data
#         # input_data = decoded_event.get('input', {})  # Use get() to avoid KeyError
#         # image_data = input.get('ImageData')  # Base64 encoded image
#         image_data_base64 = input.input.get('ImageData')
#         if not image_data_base64:
#             return {"error": "ImageData not provided in input"}

#         # Decode the base64 image data
#         image_data = base64.b64decode(image_data_base64)

#         # print("image_data =", image_data)
#         image = Image.open(BytesIO(image_data))

#         # Convert the image to OpenCV format (BGR)
#         # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

#         # Make predictions using the saved image
#         tags, predictions = make_prediction(image, saved_model=True)
#         print(f"FastAPI, predictions are {predictions}")

#         # Return predictions as JSON
#         return {
#             "predictions": predictions.tolist(),
#             "tags": tags
#         }
    
#     except Exception as e:
#         return {"error": str(e)}


# @app.post("/upload")
# async def upload(file: UploadFile = File(...)):
#     try:
#         # Asynchronously read the file content
#         contents = await file.read()
#         # Save the uploaded file
#         with open(file.filename, 'wb') as f:
#             f.write(contents)
#     except Exception as e:
#         return {"message": f"There was an error uploading the file: {e}"}
#     finally:
#         await file.close()  # Close the file asynchronously

#     return {"message": f"Successfully uploaded {file.filename}"}


# # GET test endpoint
# @app.get("/testget")
# async def test_get():
#     return {"message": "GET API is working!"}

# handler = Mangum(app)  # This is your AWS Lambda handler

if __name__ == "__main__":
    # Use this for debugging purposes only
    logger.warning("Running in development mode. Do not run like this in production mode.")
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000, log_level="debug")

# @app.post("/upload")
# async def upload(file: UploadFile = File(...)):
#     try:
#         # Asynchronously read the file content
#         contents = await file.read()
#         # Save the uploaded file
#         with open(file.filename, 'wb') as f:
#             f.write(contents)
#     except Exception as e:
#         return {"message": f"There was an error uploading the file: {e}"}
#     finally:
#         await file.close()  # Close the file asynchronously

#     return {"message": f"Successfully uploaded {file.filename}"}

# @app.get("/files/{file_name}")
# async def get_file(file_name: str):
#     file_path = os.path.join(UPLOAD_DIRECTORY, file_name)
#     if os.path.exists(file_path):
#         return FileResponse(file_path)
#     return {"error": "File not found"}
# 
#   

# # Use the loaded_model for predictions
# @app.get("/predict/{item_id}")
# async def predict(item_id: int):
#     # Replace this with your actual input data
#     input_data =????????  # Provide the input data for prediction
#     prediction = loaded_model.predict(input_data)
#     return {"item_id": item_id, "prediction": prediction.tolist()}

# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# from nn_model.config.core import config

# import numpy as np
# import pickle
# from pydantic import BaseModel

# app = FastAPI()

# path_pkl = r'C:\Course_online\Deploying-Machine-learning-models-thirdClone\nn_model\trained-models\nn_model_output_v0.0.1.pkl'

# # Load the model from the pickle file
# with open(path_pkl, 'rb') as model_file:
#     model = pickle.load(model_file)

# class Prediction(BaseModel):
#     # Define your input data model if needed
#     pass

# @app.post("/predict")
# async def predict_image(file: UploadFile = File(...), data: Prediction = None):
#     try:
#         # Save the uploaded image
#         file_path = (config.app_config.uploaded_images_dir.resolve() / file.filename).resolve()

#         with open(file_path, "wb") as image_file:
#             image_file.write(file.file.read())

#         # Use the loaded model for prediction
#         # Modify this part based on your specific data model and how you want to use the data for prediction
#         prediction_result = model.predict(data)  # Replace 'data' with your actual input data

#         # Return predictions as JSON
#         return JSONResponse(content={"predictions": prediction_result.tolist()})

#     except Exception as e:
#         return JSONResponse(content={"error": f"An error occurred: {str(e)}"}, status_code=500)


#  from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# from nn_model.processing.data_manager import load_single_image
# from nn_model.config.core import config
# from nn_model.predict import make_prediction

# import numpy as np
# from pydantic import BaseModel

# app = FastAPI()

# @app.post("/predict")
# async def predict_image(file: UploadFile = File(...)):
#     try:
#         # Save the uploaded image
#         file_path = (config.app_config.uploaded_images_dir.resolve() / file.filename).resolve()

#         # file_path = config.app_config.uploaded_images_dir / file.filename
#         with open(file_path, "wb") as image_file:
#             image_file.write(file.file.read())

#         # Make predictions using the saved image
#         predictions = make_prediction(file_path, pipeline_file_name='pipeline_nn_model_1.0.0.pkl')

#         # Return predictions as JSON
#         return JSONResponse(content={"predictions": predictions.tolist()})

#     except Exception as e:
#         return JSONResponse(content={"error": f"An error occurred: {str(e)}"}, status_code=500)

