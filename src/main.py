import time
import boto3
import requests
import streamlit as st

API_URL = "https://api-inference.huggingface.co/models/NehaBardeDUKE/autotrain-ai-generated-image-classification-3250490787"
headers = {f"Authorization": "Bearer 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"}

# Using Amazon S3
s3 = boto3.resource("s3")
s3_client = boto3.client("s3")
bucket_name = "ai-or-human-bucket"
default_region = "ap-southeast-1"


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
    with open(filename, "rb") as file:
        payload = file.read()
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
    # Upload the image to AWS S3
    s3.Bucket(bucket_name).put_object(Key=uploaded_file.name, Body=uploaded_file)

    # Construct the AWS S3 URL
    bucket_url = (
        f"https://{bucket_name}.s3.{default_region}.amazonaws.com/{uploaded_file.name}"
    )

    # Fetch the image from AWS S3
    img = s3_client.get_object(Bucket=bucket_name, Key=uploaded_file.name)
    st.write(img)

    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Query the API and store the response
    response = query(uploaded_file.name)

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
