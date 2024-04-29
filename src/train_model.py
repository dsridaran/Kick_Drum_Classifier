from data_preprocessing import prepare_data, split_data
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import cv2
import random
from data_preprocessing import prepare_data, split_data

def train_classifier(model_name = "none", model_save = True, overwrite = False, drum_path = '../data/raw/drum_samples_from_NN_training_set', kick_path = '../data/raw/kick_samples_from_NN_training_set', X_train_labs = None, x_percent = 1.0, type = "center", noise_factor = 0.0, verbose = False, epochs = 3, batch_size = 32, validation_split = 0.2, seed = 1, plot_cm = False):
    """
    Train binary classification model.

    Parameters:
    model_name (str): Name of model.
    overwrite (bool): Overwrites existing model if True.
    drum_path (str): File path to drum audio samples.
    kick_path (str): File path to kick audio samples.
    X_train_labs (list): Optional list of matches over which to train model.
    x_percent (float): Percentage of 0.4 second audio to train model.
    type (string): Section of sound audio to retain ("start", "center", "end", or "random")
    noise_factor (float): Articifical noise factor to add to raw audio.
    verbose (bool): Outputs printed if True.
    epochs (int): Number of epochs to train model.
    batch_size (int): Batch size to train model.
    validation_split (float): Validation split to evaluate model.
    seed (int): Seed for reproducability.
    plot_cm (bool): Confusion matrix printed if True.
   
    Returns:
    model: Trained model.
    """    
    # Abort if model already exists
    file_path = f'../models/{model_name}.keras'
    if os.path.exists(file_path) and model_save and not overwrite:
         return "Model already exists. Please provide a new model name or set: overwrite = True."
        
    # Extract train and test data
    X, y, X_labs = prepare_data(drum_path, kick_path, x_percent, type, noise_factor, verbose)
    X_train, X_test, y_train, y_test = split_data(X, y, X_labs, X_train_labs = X_train_labs, verbose = verbose)
    
    # Set seed for reproducability
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    # Define model
    model = Sequential([
        Conv2D(32, (3, 3), activation = 'relu', input_shape = (X_train.shape[1], X_train.shape[2], 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation = 'relu', name = 'final_layer'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation = 'relu'),
        Dropout(0.5),
        Dense(2, activation = 'softmax')
    ])

    # Train model
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, validation_split = validation_split, verbose = 0)

    # Evaluate model
    test_loss, test_acc, test_cm = evaluate_model(model, X_test, y_test)
    print(f"Test accuracy: {test_acc}, Test loss: {test_loss}")
    
    if plot_cm: 
        sns.heatmap(cm, annot = True, fmt = "d", cmap = "Blues")
        plt.xlabel('Predicted Labels'); plt.ylabel('True Labels'); plt.title('Confusion Matrix')
        plt.show() 
        
    # Save model
    if model_save:
        model.save(file_path)
        print(f"Model successfully saved: {file_path}")

    return model, test_acc

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model out of sample.

    Parameters:
    model: Trained model.
    X_test: X test data.
    y_test: y test data
    
    Returns:
    tuple: Contains multiple elements:
        - test_loss: Test loss
        - test_acc: Test accuracy
        - test_cm: Test confusion matrix
    """   

    # Calculate loss and accuracy
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose = 0)
    
    # Generate confusion matrix
    y_pred = model.predict(X_test)
    y_pred_digits = np.argmax(y_pred, axis = 1)
    y_test_digits = np.argmax(y_test, axis = 1)
    test_cm = confusion_matrix(y_test_digits, y_pred_digits)
    
    return test_loss, test_acc, test_cm   