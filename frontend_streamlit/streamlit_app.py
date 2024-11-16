import streamlit as st
import requests
import base64
import os
import sys
from PIL import Image
import json
from PIL import ImageFile
from io import BytesIO
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Dynamically import the lambda_function from the correct directory
lambda_docker_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# print("lambda_docker_path = ", lambda_docker_path)

# Append the lambda_docker directory to sys.path
sys.path.append(lambda_docker_path)

# Import the Lambda handler after modifying sys.path
# from deploy_lambda.my-lambda-function import lambda_handler, image_to_base64

# Load labels from cached_labels.txt file
labels_file_path = '../nn_model/cached_labels.txt'
with open(labels_file_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

AWS_API_GATEWAY_URL = "https://zugp55gdfa.execute-api.us-west-2.amazonaws.com"
# Streamlit App
# Local FastAPI URL
LOCAL_API_URL = "http://127.0.0.1:8000"

st.title("Image Prediction")

st.sidebar.title("Select Model Core")
backend_option = st.sidebar.selectbox("Choose Backend", ("AWS Lambda", "Local"))

# Set the API URL based on user selection
if backend_option == "AWS Lambda":
    API_URL = AWS_API_GATEWAY_URL
    st.sidebar.write("Using AWS Lambda")
else:
    API_URL = LOCAL_API_URL
    st.sidebar.write("Using Local FastAPI")

def image_to_base64(image):
    """Convert an in-memory image (PIL Image) to a Base64-encoded string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")  # You can change format based on image type
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Adding a section for the TEST_PATH endpoint
st.header("Reverse Text Endpoint (/test)")

user_input = st.text_input("Enter some text:")
if st.button("Submit"):

    if user_input:
        # Prepare payload to send to the API Gateway
        payload = {
            'input': user_input  # Pass user input to the Lambda function
        }

        headers = {
            'Content-Type': 'application/json'  # Ensure the content type is set correctly
        }

        try:
            # Send POST request to the API Gateway /test endpoint
            response = requests.post(f"{API_URL}/test", json=payload, headers=headers)

            # Check if the response status code indicates success (200)
            if response.status_code == 200:
                try:
                    # Parse the JSON response content
                    result = response.json()
                    st.write("Response from API Gateway:")
                    st.json(result)
                except json.JSONDecodeError:
                    st.error("Error: Received response is not valid JSON")
                    st.write(response.text)  # Show raw text if JSON parsing fails
            else:
                st.error(f"Error: Received status code {response.status_code}")
                st.write(response.text)  # Show error response content

        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
    else:
        st.write("Please enter some text.")

# Upload an image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Convert image to base64
    image_base64 = image_to_base64(image)

    # Create payload for POST request to /pred
    payload = {
        'ImageData': image_base64
    }

    headers = {
        'Content-Type': 'application/json'  # Ensure the content type is set correctly
    }

    # Add a button to trigger the prediction
    if st.button("Get prediction of tags"):
        st.write("Sending image data to API...")

        with st.spinner('Processing...'):
            # Send POST request to the selected backend

            # Send POST request to FastAPI /pred endpoint
            response = requests.post(f"{API_URL}/pred", json=payload, headers=headers)

        # Process the response
        if response.status_code == 200:
            try:
                result = response.json()

                # Format the tags
                tags = result.get('tags', [])
                formatted_tags = ", ".join(tags)
                st.success(f"Tags: {formatted_tags}")

                # Get predictions
                predictions = result.get('predictions', [])
                if predictions and len(predictions) > 0:
                    predictions_list = predictions[0]  # Access the first list inside 'predictions'

                    # Ensure the number of predictions matches the number of labels
                    if len(predictions_list) == len(labels):
                        # Display each label with its corresponding prediction
                        st.write("Predictions:")
                        for label, pred in zip(labels, predictions_list):
                            st.write(f"{label}: {pred}")
                    else:
                        st.error("The number of predictions does not match the number of labels.")
                else:
                    st.write("No predictions found.")

            except json.JSONDecodeError:
                st.error("Error: Response is not valid JSON.")
                st.write(response.text)
        else:
            st.error(f"Error: Received status code {response.status_code}")
            st.write(response.text)
        
# def send_image_to_lambda(image_data):
#     url = 'https://vipjigda8a.execute-api.us-west-2.amazonaws.com/pred'
#     headers = {"Content-Type": "application/json"}

#     # Base64 encode image data
#     image_base64 = base64.b64encode(image_data).decode('utf-8')

#     # Create payload
#     payload = {
#         'input': {
#             'ImageData': image_base64
#         }
#     }

#     response = requests.post(url, json=payload, headers=headers)
#     return response.json()

# st.title("Image Prediction App")
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

# if uploaded_file is not None:
#     image_data = uploaded_file.read()  # Get binary image data
#     response = send_image_to_lambda(image_data)
#     st.write(response)
