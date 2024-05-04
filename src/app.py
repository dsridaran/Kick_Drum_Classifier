import streamlit as st
import os
from call_model import predict_sounds, load_files

# Set the page configuration and layout
st.set_page_config(page_title='Kick and Drum Classifier', layout='wide')

# Function to save uploaded file
def save_uploaded_file(uploaded_file):
    try:
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path  # Return the path of the saved file
    except Exception as e:
        print(e)
        return None

# Sidebar for uploading files
with st.sidebar:
    st.title('Upload Audio')
    uploaded_file = st.file_uploader("Choose a WAV file", type=['wav'])
    if uploaded_file is not None:
        file_path = save_uploaded_file(uploaded_file)
        if file_path:
            st.success(f"File uploaded successfully: {uploaded_file.name}")
            prediction = predict_sounds(file_path, 'models/base.h5')
        else:
            st.error("Failed to save file.")
            prediction = None

# Main page content
if uploaded_file and prediction:
    st.header("Audio and Prediction Results")
    # Display the audio file
    if file_path:
        st.subheader("Raw Audio")
        st.audio(file_path, format='audio/wav', start_time=0)

    # Display the model prediction results
    if prediction:
        st.subheader("Model Prediction")
        for result in prediction:
            st.markdown(f"**Predicted Class:** {result['predicted_class']}")
            st.markdown(f"**Confidence:** {result['confidence']:.2%}")

            # Conditional image display based on the prediction
            if result['predicted_class'] == "Kick":
                st.image('images/kick.png', caption='Kick Drum')
            elif result['predicted_class'] == "Drum":
                st.image('images/drum.png', caption='Drum')
    else:
        st.error("Failed to make predictions.")
