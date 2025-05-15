# Speech Emotion Recognition using CNN and MFCC

This project aims to build a **Speech Emotion Recognition (SER)** system using a **Convolutional Neural Network (CNN)** with **Mel-Frequency Cepstral Coefficients (MFCC)** features to classify emotions from audio recordings.

---

## Table of Contents

- [Background](#background)
- [Project Objectives](#project-objectives)
- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
  - [Feature Extraction](#feature-extraction)
  - [Model Architecture](#model-architecture)
  - [Model Training](#model-training)
- [Results and Evaluation](#results-and-evaluation)
- [Conclusion and Outcome](#conclusion-and-outcome)

---

## Background

Emotion recognition from human speech has wide applications ranging from automated customer service, robotics, to mental health analysis. Deep learning methods with CNNs and MFCC features have proven effective in capturing audio characteristics relevant for emotion classification.

---

## Project Objectives

- Build a CNN model capable of recognizing and classifying emotions from audio signals.
- Use MFCC features as an informative representation of speech.
- Achieve high accuracy and good generalization across datasets.

---

## Dataset Description

The dataset consists of `.wav` audio recordings labeled with emotions, including:

- **Neutral**
- **Happy**
- **Sad**
- **Angry**
- **Fearful**
- **Disgust**
- **Surprised**

Datasets can be sourced from public repositories such as RAVDESS, TESS, CREMA-D, or internal collections.

---

## Methodology

### Feature Extraction

- Audio signals are transformed into **Mel-Frequency Cepstral Coefficients (MFCC)**.
- MFCC captures important frequency components as perceived by humans.
- Each audio file is processed into an MFCC matrix serving as model input.

### Model Architecture

- Based on a **1D Convolutional Neural Network**.
- Several `Conv1D` layers with ReLU activation and `BatchNormalization`.
- `MaxPooling1D` layers reduce dimensionality.
- `Flatten` followed by `Dense` layers with `Dropout` for regularization.
- Final output layer with `softmax` activation for multi-class classification.

### Model Training

- Data is split into training, validation, and test sets.
- Model is trained using the `Adam` optimizer and `categorical_crossentropy` loss.
- Callbacks such as `EarlyStopping` and `ReduceLROnPlateau` help prevent overfitting and improve learning.
- Batch size and epochs are set based on experiments (e.g., batch size 64, 50 epochs).

---

## Results and Evaluation

- The model achieves validation accuracy above 85%.
- Evaluation metrics include:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1 Score**
- Training loss and accuracy curves show good convergence.
- Confusion matrix analysis reveals model performance on each emotion class.

---

## Conclusion and Outcome

- The CNN-based SER system using MFCC features successfully classifies emotions with satisfying performance.
- MFCC features effectively capture voice characteristics for emotion recognition.
- This model can be applied to real-time applications such as emotional chatbots, customer call analysis, and voice-based interactive systems.
- Future work can explore data augmentation, hyperparameter tuning, and more complex architectures.

