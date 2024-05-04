#from tensorflow.keras.models import load_model
import tensorflow as tf
import librosa
import os
import numpy as np

def load_files(files = None):

    # Identify files to process
    #files = os.listdir(folder)
    #files_long = [os.path.join(folder, file) for file in files]
    
    # Initialize result
    arrays = []
    #for file in files_long:

    # Load raw audio
    audio, sr = librosa.load(files, sr = None)
    mel_spec = librosa.feature.melspectrogram(y = audio, sr = sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref = np.max)
    arrays.append(mel_spec_db)
    return np.array(arrays), files

def predict_sounds(files, model = 'models/base.keras'):
#def predict_sounds(files, model = '../models/base.keras'):
    """
    Run inference on sounds using pre-trained model

    Parameters:
    files (list): List of audio files on which to run inference.
    model (str): Filepath to pre-trained model.
    
    Returns:
    list: List with files, predicted classes, and confidence.
    """

    # Pre-process audio files
    samples, names = load_files(files = files)
    samples = samples.reshape((*samples.shape, 1))

    current_directory = os.getcwd()
    directory_contents = os.listdir(current_directory)
        
     # Load model and generate predictions
    #model = load_model(model, compile = False)
    model = tf.keras.models.load_model('models/base.keras', compile = False)
    predictions = model.predict(samples, verbose = 0)

    # Store predictions
    result = []
    for i, prediction in enumerate(predictions):
        if prediction[0] > prediction[1]:
            result.append({"file": names[i], "predicted_class": "Drum", "confidence": prediction[0]})
        else:
            result.append({"file": names[i], "predicted_class": "Kick", "confidence": prediction[1]})
    
    return result
