import streamlit as st
import joblib
import cv2
import numpy as np
from PIL import Image
from skimage.feature import hog
import pandas as pd
import plotly.express as px
import tempfile
import time

st.set_page_config(
    page_title="Sign Language Gesture Recognition",
    page_icon="🤟",
    layout="wide"
)

st.markdown("""
<style>

.main{
background:#f4f7fb;
}

h1{
text-align:center;
color:#1565C0;
}

h3{
color:#3949AB;
}

.stButton>button{
background:#1565C0;
color:white;
border-radius:10px;
height:3em;
width:100%;
font-size:18px;
}

.stMetric{
background:white;
padding:15px;
border-radius:12px;
box-shadow:0px 0px 8px rgba(0,0,0,0.15);
}

</style>
""",unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load("compressed_model.pkl")


model=load_model()

st.title("🤟 Sign Language Gesture Recognition")
st.write(
"""
Recognize **A-Z**, **0-9** and **_** using a trained
Random Forest model with **HOG Features**.
"""
)

st.sidebar.title("📂 Input")

option=st.sidebar.radio(
"Choose Input",
[
"📷 Webcam",
"🖼 Upload Image",
"🎥 Upload Video"
]
)


def extract_hog(image):

    image=cv2.resize(image,(50,50))

    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    gray=gray.astype("float32")/255.0

    features=hog(
        gray,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        block_norm="L2-Hys"
    )

    return features.reshape(1,-1)


def predict_image(image):

    features=extract_hog(image)

    prediction=model.predict(features)[0]

    probability=model.predict_proba(features)[0]

    confidence=np.max(probability)

    return prediction,confidence,probability


classes=model.classes_
image = None

# ==============================
# IMAGE UPLOAD
# ==============================

if option == "🖼 Upload Image":

    uploaded = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded is not None:

        image = Image.open(uploaded).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

# ==============================
# WEBCAM
# ==============================

elif option == "📷 Webcam":

    webcam = st.camera_input("Capture Hand Gesture")

    if webcam is not None:

        image = Image.open(webcam).convert("RGB")

        st.image(
            image,
            caption="Captured Image",
            use_container_width=True
        )

# ==============================
# VIDEO
# ==============================

elif option == "🎥 Upload Video":

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        temp = tempfile.NamedTemporaryFile(delete=False)

        temp.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp.name)

        frame_placeholder = st.empty()

        prediction_placeholder = st.empty()

        progress = st.progress(0)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        count = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pred, conf, prob = predict_image(frame)

            cv2.putText(
                frame,
                f"{pred}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

            frame_placeholder.image(
                cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

            prediction_placeholder.success(
                f"Prediction : {pred} | Confidence : {conf*100:.2f}%"
            )

            count += 1

            if total > 0:
                progress.progress(min(count/total,1.0))

            time.sleep(0.02)

        cap.release()

        st.success("Video Prediction Completed ✅")

# ==============================
# IMAGE PREDICTION
# ==============================

if image is not None:

    st.divider()

    img = np.array(image)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    with st.spinner("Predicting..."):

        prediction, confidence, probability = predict_image(img)

    st.subheader("🎯 Prediction Results")

    col1, col2 = st.columns([1, 1])

    with col1:

        st.success("Prediction Completed Successfully!")

        st.metric(
            label="🔤 Predicted Symbol",
            value=str(prediction)
        )

        st.metric(
            label="📝 Predicted Class Name",
            value=str(prediction)
        )

        st.metric(
            label="📊 Prediction Confidence",
            value=f"{confidence*100:.2f}%"
        )

    with col2:

        df = pd.DataFrame({
            "Class": classes,
            "Probability": probability * 100
        })

        df = df.sort_values(
            by="Probability",
            ascending=False
        )

        fig = px.bar(
            df.head(10),
            x="Probability",
            y="Class",
            orientation="h",
            color="Probability",
            color_continuous_scale="Turbo",
            title="Top 10 Predictions"
        )

        fig.update_layout(
            height=450,
            yaxis=dict(categoryorder="total ascending")
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    st.subheader("📋 Prediction Summary")

    st.write(f"**Predicted Symbol:** {prediction}")

    st.write(f"**Predicted Class Name:** {prediction}")

    st.write(f"**Prediction Confidence:** {confidence*100:.2f}%")

    if confidence >= 0.95:
        st.success("🟢 Very High Confidence Prediction")

    elif confidence >= 0.80:
        st.info("🟡 High Confidence Prediction")

    elif confidence >= 0.60:
        st.warning("🟠 Moderate Confidence Prediction")

    else:
        st.error("🔴 Low Confidence. Try another clearer image.")

st.markdown("---")

st.markdown(
"""
<div style='text-align:center'>

### 🤟 Sign Language Gesture Recognition using Computer Vision & Machine Learning

**Random Forest + HOG Features**

Supports:

✅ Alphabets (A-Z)

✅ Numbers (0-9)

✅ Underscore (_)

Developed using **Streamlit**, **OpenCV**, **Scikit-Learn**, and **Plotly**.

</div>
""",
unsafe_allow_html=True
)
