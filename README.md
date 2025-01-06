# Iris Tumor Detection using ResNet

This project aims to classify iris images into two categories: Tumor or Normal, using deep learning techniques. The model leverages the ResNet50 architecture with transfer learning for efficient and accurate predictions.

## Project Overview

The project uses ResNet50 for classifying iris images into two categories: Tumor and Normal. The dataset consists of labeled images of normal and tumored iris. The model is trained using these images and fine-tuned to achieve better accuracy.

### Key Components:
ResNet50: Used for image classification tasks.
PyTorch: Deep learning framework used for building and training the model.
OpenCV & PIL: For image processing and augmentation.

### Features
Image pre-processing: Histogram equalization and data augmentation (rotation, flip).
Transfer learning using pre-trained ResNet50 for better performance.
Easy-to-use interface for uploading an iris image and getting a prediction.
Web app for real-time prediction.

### Installation
Follow these steps to set up the project:

1.**Clone the repository:**

  git clone https://github.com/yourusername/IrisTumorDetect.git

2.**Install dependencies:** Make sure you have Python 3.6+ installed. Then, install the required libraries:

   pip install -r requirements.txt

### Usage
1.**Run the application:** After setting up the environment, you can run the web app:

   python app.py

2.**Access the web app:** Once the application is running, open a web browser and go to http://127.0.0.1:5000/. You can upload an image of an iris to classify it as either Normal or Tumor.

### Model
#### Architecture
**ResNet50:** A deeper model that uses residual connections for better performance on complex datasets.

**Image Preprocessing:** Images are resized, augmented, and normalized for optimal model input.
#### Training
The model is trained using a dataset of labeled iris images (Tumor or Normal).

Loss function: **CrossEntropyLoss** with class weights for imbalanced data.

Optimizer: **Adam** optimizer with learning rate scheduler.

Early stopping to avoid overfitting.
#### Results
Validation Accuracy: 90%+ (Depending on your dataset).

The model can classify new images into Tumor or Normal.

