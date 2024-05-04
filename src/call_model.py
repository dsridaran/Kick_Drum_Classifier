import tensorflow as tf
import librosa
import os
import numpy as np

def predict_sounds(files, model = 'models/base.h5'):
    """
    Run inference on sounds using pre-trained model

    Parameters:
    files (list): List of audio files on which to run inference.
    model (str): Filepath to pre-trained model.
    
    Returns:
    list: List with files, predicted classes, and confidence.
    """

    # Pre-process audio files
    samples, names = load_files(file = files)
    samples = samples.reshape((*samples.shape, 1))

    current_directory = os.getcwd()
    directory_contents = os.listdir(current_directory)
        
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

def load_files(file = None):
    arrays = []
    if isinstance(file, list):
        for f in file:
            audio, sr = librosa.load(f, sr = None)
            mel_spec = librosa.feature.melspectrogram(y = audio, sr = sr)
            mel_spec_db = librosa.power_to_db(mel_spec, ref = np.max)
            arrays.append(mel_spec_db)
        return np.array(arrays), file
    else:
        audio, sr = librosa.load(file, sr = None)
        mel_spec = librosa.feature.melspectrogram(y = audio, sr = sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref = np.max)
        arrays.append(mel_spec_db)
        return np.array(arrays), [file]