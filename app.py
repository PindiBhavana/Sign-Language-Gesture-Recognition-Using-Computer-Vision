import streamlit as st
import joblib
import numpy as np
import cv2
from PIL import Image
import mediapipe as mp
import plotly.express as px
import pandas as pd

# 🎨 Page Configuration
st.set_page_config(page_title="Sign Language Predictor", page_icon="✋", layout="wide")

# Load the Random Forest Model
@st.cache_resource
def load_model():
    # <--- Point this to your new small file and use joblib
    model = joblib.load('compressed_model.pkl')
    return model

model = load_model()

# Setup MediaPipe for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Function to extract hand landmarks (assuming your RF model expects 42 or 63 features)
def process_image(image):
    img_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                # Extract x, y (and z if your model uses it)
                landmarks.extend([lm.x, lm.y])
            return np.array(landmarks).reshape(1, -1)
    return None

# 🌈 Main UI
st.title("✋ Sign Language Gesture Predictor 🌈")
st.markdown("Upload an image or use your webcam to translate sign language gestures in real-time.")

# Sidebar for input selection
st.sidebar.header("⚙️ Input Options")
input_method = st.sidebar.radio("Choose Input:", ("🖼 Image Upload", "📷 Webcam Capture"))

image_to_predict = None

# 📷 Input Handling
if input_method == "🖼 Image Upload":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_to_predict = Image.open(uploaded_file)
        st.image(image_to_predict, caption="Uploaded Image", width=300)

elif input_method == "📷 Webcam Capture":
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        image_to_predict = Image.open(camera_image)

# 🤖 Prediction Logic
if image_to_predict is not None:
    st.markdown("---")
    st.subheader("🤖 Prediction Results")
    
    with st.spinner("Analyzing hand landmarks...") :
        features = process_image(image_to_predict)
        
        if features is not None:
            # Get prediction and probabilities
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            classes = model.classes_
            
            # Confidence is the highest probability
            max_prob = np.max(probabilities)
            
            # Layout for results
            col1, col2 = st.columns(2)
            
            with col1:
                st.success("Hand Detected!")
                st.metric(label="🔤 Predicted Symbol / Class", value=str(prediction))
                st.metric(label="📊 Prediction Confidence", value=f"{max_prob * 100:.2f}%")
            
            with col2:
                # 📈 Probability Chart using Plotly
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
                    title="Model Confidence by Class",
                    color='Probability',
                    color_continuous_scale='sunset'
                )
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)
                
        else:
            st.error("No hand detected in the image. Please try again with a clearer view of your hand.")
