import streamlit as st
import os
from call_model import predict_sounds, load_files

st.set_page_config(page_title = 'Kick and Drum Classifier', layout = 'wide')

def save_uploaded_file(uploaded_file):
    try:
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        print(e)
        return False

with st.sidebar:
    st.title('Kick & Drum Classifier')
    uploaded_file = st.file_uploader("Upload an audio file", type=['wav'])
    if uploaded_file is not None:
        if save_uploaded_file(uploaded_file):
            st.success("File uploaded successfully: {}".format(uploaded_file.name))
            file_path = os.path.join('uploads', uploaded_file.name)
            prediction = predict_sounds(file_path, 'models/base.h5')
        else:
            st.error("Failed to save file.")

if uploaded_file is not None and prediction:
    if prediction:
        for result in prediction:
            st.markdown(f"<h2>Predicted Class: {result['predicted_class']}</h2>", unsafe_allow_html = True)
            st.markdown(f"<h2>Confidence: {result['confidence']:.2%}</h2>", unsafe_allow_html = True)
    else:
        st.error("Failed to make predictions.")

    for result in prediction:
        if result['predicted_class'] == "Kick":
            st.markdown("<div style = 'text-align: center'><img src = 'images/kick.png'></div>", unsafe_allow_html = True)
        elif result['predicted_class'] == "Drum":
            st.markdown("<div style = 'text-align: center'><img src = 'images/drum.png'></div>", unsafe_allow_html = True)
