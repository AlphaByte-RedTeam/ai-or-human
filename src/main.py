import streamlit as st

st.title("AI Image Classifier")

# Store the uploaded file
uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])

# Display the uploaded image
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
