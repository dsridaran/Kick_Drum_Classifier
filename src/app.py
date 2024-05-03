import streamlit as st
import os
from call_model import predict_sounds, load_files

st.set_page_config(page_title = 'Audio Classifier', layout = 'wide')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        return False

st.title('Audio Classification App')

uploaded_file = st.file_uploader("Upload an audio file", type = ['wav'])

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        st.success("File uploaded successfully: {}".format(uploaded_file.name))
        file_path = os.path.join('../data/uploads', uploaded_file.name)
        prediction = predict_sounds(file_path, '../models/base.keras')
        if prediction:
            for result in prediction:
                st.write(f"File: {result['file']}")
                st.write(f"Predicted Class: {result['predicted_class']}")
                st.write(f"Confidence: {result['confidence']:.2%}")
        else:
            st.error("Failed to make predictions.")
    else:
        st.error("Failed to save file.")

st.write("Please upload an audio file to get started.")
