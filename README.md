![Logo](images/logo.png "Logo")

# Kick & Drum Classifier

## Description
This project provides a deep learning system for classifying audio clips as "player kicks" or "fan drums." The goal is to accurately isolate the two events to improve sound localization techniques for tracking and broadcasting.

[Try app!](https://kickdrumclassifier.streamlit.app/) Audio files for testing can be downloaded from the "data/examples" folder.

## Installation
To set up this project, clone the repository to your local machine and ensure that you have Jupyter Notebook installed to run `.ipynb` files.

```bash
git clone https://github.com/dsridaran/Kick_Drum_Classsifier.git
cd Kick_Drum_Classsifier
```

## Usage

### Model Training

Train the classifier using the following Jupyter notebook:

```bash
jupyter notebook train_classifiers.ipynb
```

Training and testing can be performed using random partitions (by default):

```bash
train_classifier(model_name = "base")
```

Or using specific matches in the train set and specific matches in the test set:

```bash
train_classifier(model_name = "base", X_train_labs = ['XXX.wav', 'XXX.wav'])
```

Models are saved by default, but this behavior can be turned off using `model_save = False`.

To understand the models, GradCAMs can be plotted using code from `src/plot_models.py`:

```bash
plot_grad_cam(X_test, y_test, n = 5, model = model, last_conv_layer_name = 'final_layer')
```

### Model Inference

Run inference using the following Jupyter notebook:

```bash
jupyter notebook call_classifers.ipynb
```

Inference for files `['XXX.wav', 'XXX.wav']` using model `'../models/base.h5'` is called using:

```bash
predict_sounds(files = ['XXX.wav', 'XXX.wav'], model = '../models/base.h5')
```

## Input Data Structure

The expected inputs for each match are organized as follows:

- **data/raw/drum/{file}.wav:** 0.4 second 48kHz audio data for drums.
- **data/raw/kick/{file}.wav:** 0.4 second 48kHz audio data for kicks.

## Contact

For questions or support on this code, please contact dilan.sridaran@gmail.com.
