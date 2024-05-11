import tensorflow as tf
import librosa
import os
import numpy as np
from data_preprocessing import augment_wav

def predict_sounds(files, model = 'models/base.h5', x_percent = 1.0, type = "center"):
    """
    Run inference on sounds using pre-trained model

    Parameters:
    files (list): List of audio files on which to run inference.
    model (str): Filepath to pre-trained model.
    x_percent (float): Percentage of 0.4 second audio to train model.
    type (string): Section of sound audio to retain ("start", "center", or "end").
    
    Returns:
    list: List with files, predicted classes, and confidence.
    """
    
    # Pre-process audio files
    samples, names = load_files(file = files, x_percent = x_percent, type = type)
    samples = samples.reshape((*samples.shape, 1))
        
    # Load model and generate predictions
    model = tf.keras.models.load_model(model)
    predictions = model.predict(samples, verbose = 0)

    # Store predictions
    result = []
    for i, prediction in enumerate(predictions):
        if prediction[0] > prediction[1]:
            result.append({"file": names[i], "predicted_class": "Drum", "confidence": prediction[0]})
        else:
            result.append({"file": names[i], "predicted_class": "Kick", "confidence": prediction[1]})
    
    return result

def load_files(file = None, x_percent = 1.0, type = "center"):
    """
    Preprocess data for model inference

    Parameters:
    files (list): List of audio files on which to run inference.
    x_percent (float): Percentage of 0.4 second audio to train model.
    type (string): Section of sound audio to retain ("start", "center", or "end").
    
    Returns:
    list: List with files, predicted classes, and confidence.
    """
    
    arrays = []
    if isinstance(file, list):
        for f in file:
            audio, sr = librosa.load(f, sr = None)
            augmented_audio = augment_wav(y = audio, x_percent = x_percent, type = type)
            mel_spec = librosa.feature.melspectrogram(y = augmented_audio, sr = sr)
            mel_spec_db = librosa.power_to_db(mel_spec, ref = np.max)
            arrays.append(mel_spec_db)
        return np.array(arrays), file
    else:
        audio, sr = librosa.load(file, sr = None)
        augmented_audio = augment_wav(y = audio, x_percent = x_percent, type = type)
        mel_spec = librosa.feature.melspectrogram(y = augmented_audio, sr = sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref = np.max)
        arrays.append(mel_spec_db)
        return np.array(arrays), [file]
