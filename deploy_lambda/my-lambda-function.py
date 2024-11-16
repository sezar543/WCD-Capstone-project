import os
import boto3
import json
import tensorflow as tf
import base64
from io import BytesIO
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



import numpy as np
import sys

# Add the parent directory to sys.path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add the 'nn_model' directory to the Python path
# sys.path.append(os.path.join(os.path.dirname(__file__), 'nn_model'))
# Add the parent directory to the sys.path to access sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn_model.predict import make_prediction
from nn_model import __version__ as _version
from nn_model.config.core import config

import logging
import time
import re
# from botocore.config import Config

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Use an environment variable for the bucket name
bucket_name = os.getenv('S3_BUCKET_NAME', 'default-bucket-name')

# config = Config(
#     region_name = 'us-west-2',
#     signature_version = 's3v4'
# )
# s3 = boto3.client('s3', config=config)

# Initialize S3 client globally for reuse
s3 = boto3.client('s3')
# Get the project root by going up one directory from the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
print("Inside lambda_function: project_root = ", project_root)
print("Inside lambda_function: current_dir = ", current_dir)
pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.h5"
relative_path_to_model = os.path.join('nn_model', 'trained-models', pipeline_file_name)
print("relative_path_to_model = ", relative_path_to_model)
# Construct the absolute path to the model file (use current_dir for docker file)
path_h5 = os.path.join(current_dir, relative_path_to_model)
print("Final path_h5 = ", path_h5)

# Declare a global variable to hold the loaded model
global_model = None

PREDICT_PATH = '/pred'
TEST_PATH = "/test"
# RETRAIN_PATH = '/retrain'  

# Mock for testing purpose
def my_load_model(mock=False):
    global global_model
    if global_model is None:
        logger.info("Loading the model for the first time...")
        model = tf.keras.models.load_model(path_h5)
        global_model = model
    # Use a local pre-saved model for testing (replace with your local model path)
    # uncomment to use local test
    # path_h5 = r"C:\Dell15\p\d-third\nn_model\trained-models\nn_model_output_v0.0.1.h5"
    else:
        logger.info("Using the cached model.")

    return global_model
    # else:
        # # S3-based model loading logic
        # prefix = 'models/'
        # paginator = s3.get_paginator('list_objects_v2')
        # result_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        # h5_files = [obj['Key'] for result in result_iterator for obj in result.get('Contents', []) if obj['Key'].endswith('.h5')]

        # # Get the latest .h5 file
        # latest_file = max(h5_files, key=lambda x: s3.head_object(Bucket=bucket_name, Key=x)['LastModified'])

        # # Download the latest model file from S3
        # obj = s3.get_object(Bucket=bucket_name, Key=latest_file)
        # model_bytes = obj['Body'].read()

        # local_model_path = f"/tmp/{latest_file.split('/')[-1]}"
        # with open(local_model_path, 'wb') as f:
        #     f.write(model_bytes)

        # model = tf.keras.models.load_model(local_model_path)
        # return model


def load_image(mock=False, image_path=None, image_url=None, image_data=None):
    if mock:
        
        logger.info("Loading mock image for local testing.")
        if image_url:
            bucket_name, image_key = extract_bucket_and_key(image_url)
            obj = s3.get_object(Bucket=bucket_name, Key=image_key)
            img_data = obj['Body'].read()
            image = Image.open(BytesIO(img_data))
        elif image_path:
            image = Image.open(image_path)
        else:
            raise ValueError("No valid image url or image path is provided.")
        return image
    else:
        if image_data:
            # Decode the Base64-encoded image string
            logger.info("Decoding Base64-encoded image data.")
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            return image
        else:
            raise ValueError("No valid image data provided.")

def extract_bucket_and_key(s3_url):
    if not isinstance(s3_url, str):
        raise ValueError(f"Invalid S3 URL: Expected a string but got {type(s3_url)}")

    # Check if it's an S3 URL
    if s3_url.startswith('s3://'):
        match = re.match(r's3://([^/]+)/(.+)', s3_url)
        if not match:
            raise ValueError(f"Invalid S3 URL format: {s3_url}")
        return match.group(1), match.group(2)

    # Check if it's an HTTPS URL for S3
    elif s3_url.startswith('https://'):
        match = re.match(r'https://([^/]+)\.s3\.[^/]+\.amazonaws\.com/(.+)', s3_url)
        if not match:
            raise ValueError(f"Invalid HTTPS S3 URL format: {s3_url}")
        return match.group(1), match.group(2)

    else:
        raise ValueError(f"Invalid URL format: {s3_url}")

def lambda_handler(event, context):
    start_time = time.time()  # Record start time
    print("event = ", event)
    try:
        # Log event details
        logger.info("Received event: %s", event)
        print("Received event print:", event)
    
        if event.get('rawPath') == TEST_PATH:
            # Parse the body for the text input
            decoded_event = json.loads(event['body'])
            input_text = decoded_event.get('input', '')  # Get the 'input' text from the request

            # Reverse the input text
            reversed_text = input_text[::-1]

            return {
                'statusCode': 200,
                'body': json.dumps({'reversed_text': reversed_text})
            }
        
        # Handle '/pred' route for prediction
        if event['rawPath'] == PREDICT_PATH:
            # Assuming the input is passed in JSON format in the body
            decoded_event = json.loads(event['body'])  # Parse the input JSON
            image_data = decoded_event.get('ImageData', {})  # Base64 encoded image, Use get() to avoid KeyError           
            # Check if ImageData is provided
            if image_data:
                logger.info("Processing image data from event.")
                image = load_image(image_data=image_data)
            else:
                raise ValueError("No valid image input provided.")
            
            # Load model and make predictions
            model = my_load_model()
            tags, predictions = make_prediction(image, model)

            response = {
                "tags": tags,
                "predictions": predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
            }

            return {
                "statusCode": 200,
                "body": json.dumps(response)
            }



        # Optional: handle other actions if needed
        elif 'rawPath' in event:
            action = event['rawPath']
            if action == 'PREDICT_PATH':
                # Handle predict action logic here
                pass
            elif action == 'TEST_PATH':
                # Handle retrain action logic here
                pass

        return {
            "statusCode": 400,
            "body": json.dumps({"message": "Invalid action"})
        }

    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"message": str(e)})
        }
    

def image_to_base64(image):
    """Convert an in-memory image (PIL Image) to a Base64-encoded string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")  # You can change format based on image type
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# def image_to_base64(image_path):
#     """Convert an image file to a Base64-encoded string."""
#     with open(image_path, "rb") as img_file:
#         base64_string = base64.b64encode(img_file.read()).decode('utf-8')
#     return base64_string



# if __name__ == "__main__":
#     print(lambda_handler({'Action': 'test', "mock": True}, None))

#     event = {
#     "Action": "predict",
#     "mock": True,
#     "ImageURL": "https://wcd-capstone.s3.us-west-2.amazonaws.com/train_30.jpg"
#     }

#     image_Path = r"C:\Dell15\p\d-third\nn_model\datasets\test-jpg\test_30.jpg"
#     event2 = {
#     "Action": "predict",
#     "mock": True,
#     "ImagePath": image_Path
#     }

#     # Convert your local image to Base64
#     image_data_base64 = image_to_base64(image_Path)
#     print(image_data_base64)
#     # Create an event with Base64 image data
#     event_with_image_data = {
#         "Action": "predict",
#         "mock": False,
#         "ImageData": image_data_base64
#     }

    # print(lambda_handler(event, None))
    # print(lambda_handler(event2, None))
    # print(lambda_handler(event_with_image_data, None))

# def lambda_handler(event, context):
#     start_time = time.time()  # Record start time

#     if os.path.exists(path_h5):
#         print(f"Model file found at: {path_h5}")
#     else:
#         print(f"Model file not found at: {path_h5}")

#     try:
#         logger.info("Received event: %s", event)
#         print("Received event print:", event)
#         # Check if this is a test mode
#         mock = event.get("mock", False)

#         # Handle '/pred' route for prediction
#         if event['rawPath'] == PREDICT_PATH:
#             # Assuming the input is passed in JSON format in the body
#             decoded_event = json.loads(event['body'])  # Parse the input JSON
#             input_data = decoded_event['input']  # Replace 'input' with your actual input key
#             prediction = predict(input_data)  # Call your prediction function
#             response = {
#                 "prediction": prediction
#             }
#             return {
#                 "statusCode": 200,
#                 "body": json.dumps(response)
#             }
#         if event['Action'] == 'predict':
#             # Check if ImageURL, ImagePath, or ImageData is provided
#             if event.get('ImageData'):
#                 image_data = event['ImageData']
#                 logger.info("Processing image data from event.")
#                 # Load image from Base64-encoded string
#                 image = load_image(mock=mock, image_data=image_data)
#             elif event.get('ImageURL'):
#                 image_url = event['ImageURL']
#                 logger.info(f"image_url: {image_url}")
#                 # Load image from URL
#                 image = load_image(mock=mock, image_url=image_url)
#             elif event.get('ImagePath'):
#                 image_path = event['ImagePath']
#                 logger.info(f"image_path: {image_path}")
#                 # Load image from a path (could be S3 or local path)
#                 image = load_image(mock=mock, image_path=image_path)
#             else:
#                 raise ValueError("No valid image input provided.")
            
#             model = my_load_model(mock=mock)

#             # Make predictions (Assuming make_prediction is already defined)
#             tags, predictions = make_prediction(image, model)

#             response = {
#                 "tags": tags,
#                 "predictions": predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
#             }

#             end_time = time.time()  # Record end time
#             duration = end_time - start_time
#             print(f"Execution time: {duration} seconds")
#             return {
#                 'statusCode': 200,
#                 'body': json.dumps(response)
#             }

#         elif event['Action'] == 'test':
#             # Simple test response
#             response = {
#                 "tags": "test_tags",
#                 "predictions": "test_predictions"
#             }

#             end_time = time.time()  # Record end time
#             duration = end_time - start_time
#             print(f"Execution time: {duration} seconds")
#             return {
#                 'statusCode': 200,
#                 'body': json.dumps(response)
#             }



#         end_time = time.time()  # Record end time
#         duration = end_time - start_time
#         print(f"Execution time: {duration} seconds")        
#         # Invalid action
#         return {
#             'statusCode': 400,
#             'body': json.dumps({"message": "Invalid action specified."})
#         }

#     except Exception as e:
#         logger.error("Error occurred: %s", str(e), exc_info=True)
#         end_time = time.time()  # Record end time
#         duration = end_time - start_time
#         print(f"Execution time: {duration} seconds")
#         return {
#             'statusCode': 500,
#             'body': json.dumps({"message": str(e)})
#         }