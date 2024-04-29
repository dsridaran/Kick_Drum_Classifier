# Kick_Drum_Classifier

## Description
This project provides a deep learning system for classifying audio clips as "player kicks" or "fan drums." The goal is to accurately isolate the two events to improve sound localization techniques for tracking and broadcasting.

## Installation
To set up this project, clone the repository to your local machine and ensure that you have Jupyter Notebook installed to run `.ipynb` files.

```bash
git clone https://github.com/dsridaran/Kick_Drum_Classsifier.git
cd Kick_Drum_Classsifier
```

## Usage

Train the classifier using the following Jupyter notebook:

```bash
jupyter notebook kick_drum_classifier.ipynb
```

Training and testing can be performed using random partitions:

```bash
train_classifier()
```

Or using specific matches in the train set and specific matches in the test set:

```bash
train_classifier(X_train_labs = ['Drum_.wav', 'SFC_CFC_Kick_.wav'])
```

## Input Data Structure

The expected inputs for each match are organized as follows:

- **data/raw/{drum_folder}/{file}.wav:** 0.4 second 48kHz audio data for drums.
- **data/raw/{kick_folder}/{file}.wav:** 0.4 second 48kHz audio data for kicks.

## Contact

For questions or support, please contact dilan.sridaran@gmail.com.
