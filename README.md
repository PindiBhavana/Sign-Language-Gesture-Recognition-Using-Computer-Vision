🤟 Sign Language Gesture Recognition using Computer Vision and Machine Learning
📌 Overview

This project is a Computer Vision and Machine Learning application that recognizes sign language gestures representing English alphabets (A–Z), digits (0–9), and the underscore (_) symbol. The system predicts the corresponding gesture from an uploaded image, webcam capture, or video input and displays the predicted symbol along with its confidence score.

The model was trained on a dataset containing 55,522 sign language gesture images across 37 classes. Images were preprocessed using grayscale conversion, normalization, resizing, and Histogram of Oriented Gradients (HOG) feature extraction. Multiple machine learning algorithms were evaluated, and Random Forest achieved the best performance after hyperparameter tuning with GridSearchCV.

The application is deployed using Streamlit Community Cloud, providing a simple and interactive web interface for real-time gesture recognition.

✨ Features
🤟 Recognizes 37 sign language classes
A–Z
0–9
Underscore (_)

🖼️ Image Upload Prediction
📷 Webcam Image Prediction
🎥 Video Upload Prediction
📊 Prediction Confidence Score
📈 Interactive Probability Chart
🎨 Modern and User-Friendly Streamlit Interface

🛠️ Technologies Used

Python
OpenCV
Scikit-learn
Scikit-image (HOG)
NumPy
Pandas
Plotly
Joblib
Streamlit

🔄 Machine Learning Workflow

Data Collection
Image Preprocessing
HOG Feature Extraction
Train-Test Split (80:20)
Model Training
Hyperparameter Tuning (GridSearchCV)
Model Evaluation
Model Serialization (Joblib)
Streamlit Deployment

📊 Model

Algorithm: Random Forest Classifier

Feature Extraction: Histogram of Oriented Gradients (HOG)
Hyperparameter Tuning: GridSearchCV
Input: 50 × 50 grayscale image
Output: Predicted gesture with confidence score

📷 Application Preview

The application supports:

Uploading gesture images
Capturing images using a webcam
Uploading short gesture videos
Displaying prediction confidence
Visualizing prediction probabilities

🚀 Future Enhancements

Real-time webcam video recognition
Word and sentence formation from continuous gestures
Deep Learning (CNN) implementation
Mobile application deployment
Multi-hand gesture recognition

👩‍💻 Author

Pindi Bhavana

⭐ If you found this project useful, consider giving the repository a Star on GitHub.
