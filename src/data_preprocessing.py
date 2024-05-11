import os
import numpy as np
import re
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import soundfile as sf
from collections import Counter
import cv2
import random

def prepare_data(drum_path, kick_path, x_percent = 1.0, type = "center", verbose = False):
    """
    Preprocess sound data with augmentations (if required).

    Parameters:
    drum_path (str): File path to drum audio samples.
    kick_path (str): File path to kick audio samples.
    x_percent (float): Percentage of 0.4 second audio to train model.
    type (string): Section of sound audio to retain ("start", "center", or "end")
    verbose (bool): Outputs printed if True.
   
    Returns:
    tuple: Contains multiple elements:
        - X (array): Array of sound files.
        - y (array): Array of true labels.
        - X_labs (array): Array of sound match sources.
    """

    # Load kick and drum samples (with augmentations)
    drum_samples, drum_files = load_files(folder = drum_path, x_percent = x_percent, type = type)
    kick_samples, kick_files = load_files(folder = kick_path, x_percent = x_percent, type = type)
    X = np.concatenate((drum_samples, kick_samples), axis = 0)
    X = X.reshape((*X.shape, 1))
    
    # Identify source matches
    X_labs = np.concatenate((drum_files, kick_files), axis = 0)
    def remove_numbers(filename):
        return re.sub(r'\d+', '', filename)
    X_labs = np.vectorize(remove_numbers)(X_labs)
    
    # Create labels
    drum_labels = np.zeros(drum_samples.shape[0])
    kick_labels = np.ones(kick_samples.shape[0])
    y = np.concatenate((drum_labels, kick_labels), axis = 0)
    encoder = LabelEncoder()
    y = to_categorical(encoder.fit_transform(y))

    return X, y, X_labs

def load_files(folder = None, files = None, x_percent = 1.0, type = "center"):
    """
    Load sound files and apply augmentations.

    Parameters:
    folder (str): File path to audio samples.
    files (list): Optional list of specific files to process.
    x_percent (float): Percentage of 0.4 second audio to train model.
    type (string): Section of sound audio to retain ("start", "center", "end", or "random")
    
    Returns:
    tuple: Contains multiple elements:
        - Array of audio files (MEL spectrogram)
        - Array of source file names
    """
    
    # Identify files to process
    if files is None:
        files = os.listdir(folder)
        files_long = [os.path.join(folder, file) for file in files]
    
    # Initialize result
    arrays = []
    for file in files_long:

        # Load raw audio
        audio, sr = librosa.load(file, sr = None)
        
        # Perform augmentations
        augmented_audio = augment_wav(y = audio, x_percent = x_percent, type = type)
        
        # Convert to MEL spectrogram representation
        mel_spec = librosa.feature.melspectrogram(y = augmented_audio, sr = sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref = np.max)
        arrays.append(mel_spec_db)
    return np.array(arrays), files
    
def augment_wav(y, x_percent, type = "center"):
    """
    Augment wave with snipping and random noise.

    Parameters:
    y (array): Audio signal of sound.
    x_percent (float): Percentage of 0.4 second audio to train model.
    type (string): Section of sound audio to retain ("start", "center", or "end")
    
    Returns:
    array: Augmented sound representation.
    """

    # Calculate required number of samples
    x_samples = int(len(y) * x_percent)

    # Slice clip and add random noise
    if type == "start":
        return y[:x_samples]
    elif type == "end":
        return y[-x_samples:]
    elif type == "center":
        mid_start = len(y) // 2 - x_samples // 2
        return y[mid_start:(mid_start + x_samples)]
    else:
        return y
    
def split_data(X, y, X_labs, X_train_labs = None, test_size = 0.25, random_state = 1, verbose = False):
    """
    Partition train and test data.

    Parameters:
    X (array): Array of sound files.
    y (array): Array of true labels.
    X_labs (array): Array of sound match sources.
    X_train_labs (list): Optional list of matches over which to train model.
    test_size (float): Proportion of observations to randomly assign to test set if X_train_labs is None.
    random_state (int): Random seed for reproducability.
    verbose (bool): Distribution of outcome variable printed if true.
    
    Returns:
    tuple: Contains multiple elements:
        - X_train
        - y_train
        - X_test
        - y_test
    """

    # Partition data randomly if X_train_labs is None
    if X_train_labs is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
    else:
    # Partition data by X_train_labs
        train_mask = np.isin(X_labs, X_train_labs)
        test_mask = ~np.isin(X_labs, X_train_labs)
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
    # Print distributions (if required)
    if verbose:
        print(f"Training set distribution: {Counter(np.argmax(y_train, axis = 1))}")
        print(f"Testing set distribution: {Counter(np.argmax(y_test, axis = 1))}")
    return X_train, X_test, y_train, y_test
