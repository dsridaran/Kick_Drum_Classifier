import streamlit as st
import os
from call_model import predict_sounds, load_files

st.set_page_config(page_title = 'Kick and Drum Classifier', layout = 'wide')

def save_uploaded_file(uploaded_file):
    try:
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok = True)
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        print(e)
        return None

with st.sidebar:
    st.title('Upload Audio')
    uploaded_file = st.file_uploader("Choose a WAV file", type = ['wav'])
    if uploaded_file is not None:
        file_path = save_uploaded_file(uploaded_file)
        if file_path:
            st.success(f"File uploaded successfully: {uploaded_file.name}")
            prediction = predict_sounds(file_path, 'models/base.h5')
        else:
            st.error("Failed to save file.")
            prediction = None

if uploaded_file and prediction:
    st.header("Kick & Drum Classifier")
    if file_path:
        st.subheader("Raw Audio")
        st.audio(file_path, format = 'audio/wav', start_time = 0)

    if prediction:
        st.subheader("Model Prediction")
        for result in prediction:
            st.markdown(f"**Predicted Class:** {result['predicted_class']}")
            st.markdown(f"**Confidence:** {result['confidence']:.2%}")
            if result['predicted_class'] == "Kick":
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image('images/kick.png', width = 500)
            elif result['predicted_class'] == "Drum":
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image('images/drum.png', width = 500)
    else:
        st.error("Failed to make predictions.")
