import streamlit as st
import joblib
import numpy as np
import cv2
from PIL import Image
import mediapipe as mp
import plotly.express as px
import pandas as pd
import tempfile
import time

# 🎨 Page Configuration
st.set_page_config(page_title="Sign Language Predictor", page_icon="✋", layout="wide")

# 🌈 Custom CSS for a beautiful UI
st.markdown("""
    <style>
    .main {background-color: #FAFAFA;}
    h1 {color: #FF4B4B; text-align: center;}
    .stAlert {border-radius: 15px;}
    </style>
""", unsafe_allow_html=True)

# Load the Random Forest Model
@st.cache_resource
def load_model():
    model = joblib.load('compressed_model.pkl')
    return model

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error("Model not found. Please ensure 'compressed_model.pkl' is uploaded.")
    model_loaded = False

# Setup MediaPipe purely for drawing visual tracking lines on screen
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# 🤖 Fixed Feature Processing (Converts image to 900 features)
def process_image(image_array):
    # 1. Convert to grayscale and resize to 30x30 to get exactly 900 features (30 * 30 = 900)
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (30, 30))
    features = resized.flatten().reshape(1, -1)
    
    # 2. Run MediaPipe just to get visual landmarks to draw on the screen for the user
    img_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    hand_landmarks = None
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
    return features, hand_landmarks

# 🌈 Main UI
st.title("✋ Sign Language Gesture Predictor 🌈")
st.markdown("Translate sign language gestures using AI! Upload an image, take a picture, or process a video.")

if model_loaded:
    st.sidebar.header("⚙️ Input Options")
    input_method = st.sidebar.radio("Choose Input Method:", 
                                    ("🖼️ Image Upload", "📷 Webcam Photo", "🎥 Video Upload"))

    image_to_predict = None

    # 1. IMAGE UPLOAD
    if input_method == "🖼️ Image Upload":
        uploaded_file = st.file_uploader("Upload a clear image of a hand sign...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Force image to RGB to prevent channel errors
            image_to_predict = Image.open(uploaded_file).convert('RGB')
            st.image(image_to_predict, caption="Uploaded Image", width=400)

    # 2. WEBCAM PHOTO
    elif input_method == "📷 Webcam Photo":
        st.info("Allow camera access to take a picture of your hand gesture.")
        camera_image = st.camera_input("Take a picture 📸")
        if camera_image is not None:
            # Force image to RGB to prevent channel errors
            image_to_predict = Image.open(camera_image).convert('RGB')

    # 3. VIDEO UPLOAD
    elif input_method == "🎥 Video Upload":
        st.info("Upload a short video of a sign language gesture. The AI will process it frame-by-frame.")
        uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
        
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_video.read())
            
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty() 
            pred_text = st.empty() 
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                features, hand_landmarks = process_image(frame)
                
                if hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB", width="stretch")
                
                # Predict using the 900 features
                prediction = model.predict(features)[0]
                pred_text.markdown(f"<h2 style='text-align: center; color: #4CAF50;'>🔤 Predicted: {prediction}</h2>", unsafe_allow_html=True)
                
                time.sleep(0.04)
                
            cap.release()
            st.success("Video processing complete! 🎬")

    # 🤖 PREDICTION LOGIC (For Images/Photos)
    if image_to_predict is not None and input_method != "🎥 Video Upload":
        st.markdown("---")
        st.subheader("🤖 Prediction Results")
        
        with st.spinner("Processing image pixels..."):
            # Convert PIL RGB format to standard OpenCV BGR format
            image_array = np.array(image_to_predict)
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            features, _ = process_image(image_bgr)
            
            # Predict and calculate probabilities
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            classes = model.classes_
            max_prob = np.max(probabilities)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success("✅ Image Processed Successfully!")
                st.metric(label="🔤 Predicted Symbol / Class", value=str(prediction))
                st.metric(label="📊 Prediction Confidence", value=f"{max_prob * 100:.2f}%")
            
            with col2:
                df_probs = pd.DataFrame({
                    'Class': classes,
                    'Probability': probabilities * 100
                })
                df_probs = df_probs.sort_values(by='Probability', ascending=True)
                
                fig = px.bar(
                    df_probs, 
                    x='Probability', 
                    y='Class', 
                    orientation='h',
                    title="🌈 Model Confidence by Class",
                    color='Probability',
                    color_continuous_scale='sunset'
                )
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)
