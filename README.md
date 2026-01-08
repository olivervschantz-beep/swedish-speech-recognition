# Swedish Speech Recognition & Pronunciation Scorer

## Description
This project is an end-to-end speech recognition system designed to classify specific Swedish words and evaluate pronunciation quality. The system prompts users to speak one of three target words ("kräftskiva", "sju", "korsord") and uses deep learning to identify the word and score the user's pronunciation against a trained model.

The application includes a web-based user interface built with Streamlit, providing real-time feedback on both word classification and pronunciation accuracy.

## Technology Stack
* **Language:** Python
* **Deep Learning:** PyTorch
* **Audio Processing:** Librosa, Sounddevice, Numpy
* **Interface:** Streamlit
* **Architecture:**
    * **CNN (Convolutional Neural Network):** Used for word classification.
    * **Convolutional Autoencoder:** Used for unsupervised pronunciation scoring by measuring reconstruction error (MSE).

## Key Features
* **Audio Pipeline:** Captures raw audio, converts it to 16kHz mono `.wav` format, and extracts Log-Mel Spectrogram features.
* **Word Classification:** Identifies which of the three target words was spoken with high accuracy in controlled environments.
* **Pronunciation Assessment:** Calculates a Mean Squared Error (MSE) score to determine if the pronunciation matches the training distribution of native speakers.
* **Interactive UI:** A browser-based dashboard where users can record audio and view results immediately.

## Project Structure
The project is organized into four main modules:
* `data/`: Scripts for converting raw audio and creating datasets (`dataset_word.py`, `dataset_pronunciation.py`).
* `models/`: PyTorch definitions for the CNN classifier and Autoencoder scorer.
* `training/`: Training loops for both models, including data augmentation (noise injection, pitch shift).
* `evaluation/`: Scripts to evaluate model performance and visual confusion matrices.

## Challenges & Learnings
* **Noise Sensitivity:** The model performs well in quiet environments but struggles in noisy settings like classrooms.
* **Data Augmentation:** Learned how to implement pitch shifting and noise injection to improve model robustness.
* **Real-world Application:** Gained experience deploying ML models into a user-friendly interface using Streamlit, bridging the gap between raw code and a usable product.
