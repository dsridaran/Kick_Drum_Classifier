from tensorflow.keras.models import load_model
from data_preprocessing import load_files

def predict_sounds(files, model = '../models/base.keras'):
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

    # Load model and generate predictions
    model = load_model(model)
    predictions = model.predict(samples, verbose = 0)

    # Store predictions
    result = []

    for i, prediction in enumerate(predictions):
        if prediction[0] > prediction[1]:
            result.append({"file": names[i], "predicted_class": "Drum", "confidence": prediction[0]})
        else:
            result.append({"file": names[i], "predicted_class": "Kick", "confidence": prediction[1]})
    
    return result