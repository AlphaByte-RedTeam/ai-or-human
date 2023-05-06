import cv2
import time
import requests
import numpy as np
import streamlit as st

API_URL = "https://api-inference.huggingface.co/models/NehaBardeDUKE/autotrain-ai-generated-image-classification-3250490787"
headers = {f"Authorization": "Bearer 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"}


def query(filename):
    """
    This function reads a file, sends its contents as a payload in a POST request to an API URL, and
    returns the response in JSON format.

    :param filename: The name of the file that contains the data to be sent in the API request
    :return: the JSON response from a POST request made to an API endpoint, after reading the contents
    of a file specified by the `filename` parameter and using it as the payload for the request. If
    there is a timeout or connection error during the request, the function will return an error message
    instead.
    """
    _, buffer = cv2.imencode(".jpg", filename)
    payload = buffer.tobytes()
    
    try:
        res = requests.post(API_URL, headers=headers, data=payload, timeout=5)
        st.success("Classification successful!", icon="✅")
    except requests.Timeout:
        st.error("Timeout error. Please try again.", icon="❌")
        return
    except requests.ConnectionError:
        st.error("Connection error. Please try again.", icon="❌")
        return
    return res.json()


st.set_page_config(
    page_title="AI Image Classifier",
    page_icon="🤖",
)


st.title("AI Image Classifier")

# Store the uploaded file
uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])

# Display the uploaded image
if uploaded_file is not None:
    # Convert file to  an opencv image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_img = cv2.imdecode(file_bytes, 1)

    st.image(
        opencv_img, caption="Uploaded Image.", use_column_width=True, channels="BGR"
    )

    # Query the API and store the response
    response = query(opencv_img)

    time.sleep(1)

    # Get the value from the response
    human_label = response[0]["label"]
    human_score = response[0]["score"] * 100

    artificial_label = response[1]["label"]
    artificial_score = response[1]["score"] * 100

    # Display the percentage of the uploaded image
    human_score_bar = st.progress(int(human_score), text=f"Human: {human_score:.3f}%")
    artificial_score_bar = st.progress(
        int(artificial_score), text=f"Artificial {artificial_score:.3f}%"
    )
