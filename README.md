# eye-wear
AI-Powered Virtual Eyewear Recommender
This project is an intelligent virtual eyewear try-on system that uses AI to recommend and visualize glasses on a user's face. It features two primary modes: a real-time webcam application for a live try-on experience and a web application built with Flask for trying on glasses using an uploaded image.

The core of the project is a Convolutional Neural Network (CNN), trained from scratch, to identify a user's face shape (Heart, Oblong, Oval, Round, or Square). Based on this prediction, the system suggests eyewear styles best suited for the user, solving the common problem of "what glasses look good on me?"

(You should add a GIF of your project in action here! A screen recording of the webcam app or the web app would be very effective.)

<hr>

## 🎯 The Problem It Solves
Shopping for eyewear online is challenging. Customers often face the "imagination gap"—they can't be sure how a pair of glasses will actually look on their unique face. This uncertainty leads to:

Purchase Hesitation: Potential customers are stopped from buying.

High Return Rates: Products that don't meet expectations are sent back, which is costly for businesses.

Lack of Personalization: Standard e-commerce sites offer little to no personalized styling advice.

This project directly tackles these issues by creating an interactive and personalized shopping experience, boosting user confidence and making the process of finding the perfect pair of glasses fun and easy.

<hr>

## ✨ Core Features
AI Face Shape Detection: A custom-trained CNN model classifies faces into five distinct shapes with high accuracy.

Personalized Recommendations: Automatically suggests eyewear styles that are stylistically proven to complement the detected face shape.

Real-time Webcam Try-On: Uses OpenCV and dlib to detect facial landmarks and overlay selected glasses onto the user's face live via their webcam.

Web-Based Image Try-On: A user-friendly Flask application allows users to upload a photo, get their face shape analyzed, and see the recommended glasses on their picture.

<hr>

## 🛠️ Technology Stack
Machine Learning & Backend: Python, TensorFlow (Keras), OpenCV, Dlib, Scikit-learn, NumPy, Flask

Frontend: HTML/CSS (via Flask Templates)

Core Model: A Convolutional Neural Network (CNN) built with Conv2D, MaxPooling2D, BatchNormalization, and Dropout layers to ensure robust and accurate predictions.

Dataset: Trained on a custom FaceShape_Dataset containing thousands of labeled images.

<hr>

## ⚙️ How It Works
The project pipeline is broken down into two main parts: Model Training and Application Deployment.

Model Training (faceShape_Main.py)

Data Loading & Preprocessing: Images are loaded from the dataset, resized to 32x32 pixels, and converted to grayscale. Histogram equalization is applied to normalize lighting.

Data Augmentation: ImageDataGenerator is used to create more robust training data by applying random rotations, shifts, zooms, and flips.

Training: The CNN model is trained on the augmented dataset using callbacks like EarlyStopping and ReduceLROnPlateau to optimize performance and prevent overfitting.

Saving: The final trained model is saved as faceshape_model.h5.

Application (Webcam & Web App)

Face Detection: An input image (from webcam or upload) is processed to detect a face using dlib's frontal face detector.

Shape Prediction: The detected face is cropped, preprocessed (the same way as the training data), and fed into the loaded faceshape_model.h5 to get a shape prediction (e.g., "Oval").

Landmark Detection: Dlib's shape_predictor_68_face_landmarks.dat is used to pinpoint the location of the eyes on the detected face.

Overlay Logic: Based on the predicted face shape, a suitable pair of glasses is selected from the Resources folder. The glasses' image is resized based on the distance between the user's eyes and then overlaid onto the original image or video frame.

<hr>

## 🚀 Setup and Installation
To run this project locally, follow these steps:

faceshape dataset from kaggle:https://www.kaggle.com/datasets/zeyadkhalid/faceshape-processed

Clone the repository:

Bash

git clone https://github.com/Shan20105/eye-wear.git
cd your-repo-name
Set up a Python virtual environment:

Bash

python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
Install the required libraries:Create a requirements.txt file and then run:

Bash

pip install -r requirements.txt
Key libraries include tensorflow, opencv-python, dlib, flask, numpy, scikit-learn, imutils.

Download the dlib Landmark Model:

You must have the shape_predictor_68_face_landmarks.dat file in the root directory. You can download it from the dlib website.

Run the Application:

For the Real-time Webcam Demo:

Bash

python faceShapeTest.py
For the Web Application:

Bash

python app.py
Then, open your browser and go to http://127.0.0.1:5000.
