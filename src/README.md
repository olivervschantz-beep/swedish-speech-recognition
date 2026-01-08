# Source Code Documentation

This directory contains the core machine learning implementation and data processing pipelines for the Swedish Speech Recognition project.

## Directory Structure

### 1. `data/`
Handles data preprocessing, loading, and transformation.
* **`dataset_word.py`**: PyTorch Dataset class for the word classifier. Handles loading audio and mapping labels.
* **`dataset_pronunciation.py`**: PyTorch Dataset class for the pronunciation autoencoder.
* **`convert_to_wav.py`**: Utility script to standardize raw audio inputs (converts to 16kHz mono `.wav`).
* **`transforms.py`**: Implements audio feature extraction (Log-Mel Spectrograms) and data augmentation pipelines.

### 2. `models/`
Contains the neural network architecture definitions.
* **`word_classifier.py`**: A Convolutional Neural Network (CNN) designed to classify the input audio into one of three target words ("kräftskiva", "sju", "korsord").
* **`pronunciation_scorer.py`**: A Convolutional Autoencoder. This model learns the latent representation of "correct" pronunciations and calculates a reconstruction error to score new inputs.

### 3. `training/`
Scripts for training the models.
* **`train_word_classifier.py`**: Training loop for the CNN, including loss calculation (CrossEntropy) and backpropagation.
* **`train_pronunciation.py`**: Training loop for the Autoencoder, optimizing for minimal reconstruction error (MSE).

### 4. `evaluation/`
Scripts for validating model performance against test data.
* **`evaluate_word_classifier.py`**: Calculates accuracy and generates confusion matrices to visualize classification errors.
* **`evaluate_pronunciation.py`**: Computes the Mean Squared Error (MSE) on test sets to verify that incorrect pronunciations yield higher error scores than correct ones.

### 5. `inference/`
* **`predict.py`**: The main inference pipeline. It accepts a raw `.wav` file, processes it through the trained models, and outputs both the predicted word and a pronunciation grade.

---
*Note: Run `pip install -r ../requirements.txt` from the root directory to ensure all dependencies are installed before running these scripts.*
