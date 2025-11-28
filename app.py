# -------------------- IMPORTS --------------------
import streamlit as st                      
from PIL import Image                       
import matplotlib.pyplot as plt             
import random                              
import tempfile                             # For temporary file handling
import joblib                               # To load classical ML models 
import pickle                               # To load tokenizer
import time                                 # To simulate delays / loading spinner
from preprocessing import preprocess_new_texts, clean_text  # Custom preprocessing functions
from tensorflow.keras.models import load_model             # Load deep learning model
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Pad sequences for NN input
import cv2
from PIL import Image
import numpy as np


# -------------------- CONFIG --------------------
st.set_page_config(page_title="Emotion Detection", page_icon="🧠", layout="wide") #page_title: The browser tab title 


# -------------------- LOAD MODELS --------------------
# Cache the ML model to avoid reloading every time user interacts
@st.cache_resource
def load_ml_model():
    return joblib.load("ensemble_model.pkl")

# Load your pipeline
model = load_ml_model()

# Cache the Neural Network model + tokenizer
@st.cache_resource
def load_nn_model():
    model = load_model("model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

nn_model, nn_tokenizer = load_nn_model()

MAX_LEN = 75  # same as used in your model training

@st.cache_resource
def load_cnn_model():
    return load_model("cnn_model.keras")   # or cnn_model.h5 depending on what you saved

cnn_model = load_cnn_model()
# -----------------------------
# Prediction Function
# -----------------------------
# ---------- Label mapping ----------
label_map = {
    "happy": 0,
    "neutral": 1,
    "sad": 2,
    "fearful": 3,
    "angry": 4,
    "surprised": 5,
    "disgusted": 6
}

# Invert mapping (num → class name)
inv_label_map = {v: k for k, v in label_map.items()}

# ---------- Prediction function ----------
def predict_emotion(model, img_path):
    # Load and preprocess image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # grayscale
    img = cv2.resize(img, (48, 48))                  # resize to 48x48
    img = img.astype("float32") / 255.0              # normalize 0-1
    img = np.expand_dims(img, axis=-1)               # add channel dim
    img = np.expand_dims(img, axis=0)                # add batch dim

    # Predict
    probs = model.predict(img)[0]  # shape (7,)
    
    # Print probabilities for each class
    for i, p in enumerate(probs):
        print(f"{inv_label_map[i]}: {p:.4f}")

    # Final prediction
    pred_idx = np.argmax(probs)
    pred_label = inv_label_map[pred_idx]
    print(f"\nFinal Prediction: {pred_label} ({probs[pred_idx]:.4f})")

    return pred_label, probs



def preprocess_for_nn(text):
    text = clean_text(text)
    seq = nn_tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    return padded

# -------------------- SESSION STATE FOR NAVIGATION --------------------
# Remember which page user is on between reruns
if "page" not in st.session_state:
    st.session_state.page = "Home Page"

# -------------------- SIDEBAR --------------------
st.sidebar.title("Navigation")
choice = st.sidebar.radio(
    "Go to:",
    ["Home Page", "Text Analysis", "Image Analysis"],
    index=["Home Page", "Text Analysis", "Image Analysis"].index(st.session_state.page)
)

# If user selects a different page, update session state and rerun
if choice != st.session_state.page:
    st.session_state.page = choice
    st.rerun()

# -------------------- HOME PAGE --------------------
if st.session_state.page == "Home Page":
    st.markdown("<h1 style='text-align: center;'>🧠 Emotion Detection App</h1>",unsafe_allow_html=True)
    st.subheader("📝 Understand your emotions through text & image")

    st.write("""
    Welcome to the **Emotion Detection App**!  
    This tool helps you analyze emotions from:
    - 📝 **Text** — Detects emotions like joy, sadness, anger, etc.
    - 🖼 **Images** — Recognizes facial emotions.
    
    ---
    ### 🔹 Features
    - Real-time emotion detection from your text input.
    - Facial emotion recognition using AI.
    - Motivational quotes for negative moods.
    
    ---
    ### 📖 How to Use
    1. **Choose** a mode from the sidebar.
    2. **Text Analysis**: Enter your thoughts and analyze emotions.
    3. **Image Analysis**: Upload a clear face photo.
    """)

    st.info("💡 Example: Enter 'I feel so happy today!' → Detected Emotion: **Joy** 🎉")

    st.markdown("### 🎭 Supported Emotions")
    st.write("😊 Joy | 😢 Sadness | 😡 Anger | 😲 Surprise | 😨 Fear | 💖 Love")

    # Quick navigation buttons
    if st.button("📝 Try Text Analysis"):
        st.session_state.page = "Text Analysis"      
        st.rerun()
    if st.button("🖼 Try Image Analysis"):
        st.session_state.page = "Image Analysis"
        st.rerun()
    
    st.success("✅ Tip: This app does not store any data — your privacy is safe.")

    st.markdown("---")
    st.subheader("💡 Why Emotion Detection Matters")
    st.write("""
        Understanding emotions can have real-life applications such as:
        - 🧘 **Mental health check-ins**: Reflect on your feelings and improve well-being.
        - 📱 **Social media sentiment tracking**: Analyze public opinion and trends in real-time.
    """)

# -------------------- TEXT ANALYSIS --------------------
if st.session_state.page == "Text Analysis":
    st.header("📝 Text Emotion Analysis")
    user_text = st.text_area("💬 How are you feeling today? Share your thoughts below:")

    # Model selection (NN or Classical ML)
    model_choice = st.radio(
        "⚙️ Choose a model for analysis:",
        ["Neural Network (Deep Learning)", "Classical ML (Ensemble Learning (LogRegr, SVM, NB))"]
    )

    if st.button("Analyze Text"):
        category_colors = {
                        0: '#89CFF0',  # sadness
                        1: '#B19CD9',  # joy
                        2: '#FFB6C1',  # love
                        3: '#FF8C42',  # anger
                        4: '#FFF176',  # fear
                        5: '#77DD77'   # surprise
        }
        if user_text.strip() == "":
            st.warning("⚠️ Please enter some text.")
        else:
            if model_choice == "Neural Network (Deep Learning)":
                st.success("✅ Using Neural Network model for emotion detection...")
                with st.spinner("Analyzing your data"):
                    time.sleep(2)
                    preprocessed_text = preprocess_for_nn(user_text)
                    probs = nn_model.predict(preprocessed_text)[0]
                    pred_class = probs.argmax()
                    categories = {0:"sadness", 1:"joy", 2:"love", 3:"anger", 4:"fear", 5:"surprise"}
                    # Original text
                    st.write("Original: ",user_text)
                    # Predicted emotion with colored background
                    predicted_emotion = categories[pred_class]
                    color = category_colors[pred_class]
                    st.markdown(f"<p style='font-size:18px; font-weight:bold; color:white; " f"background-color:{color}; padding:6px; border-radius:5px;'>"  f"🎯 Predicted Emotion: {predicted_emotion}</p>",unsafe_allow_html=True
                        )   
                    # Probabilities text                      
                    prob_text = ", ".join([f"{categories[j]}:{probs[j]:.2f}" for j in range(len(probs))])
                    st.write("Probabilities: ",prob_text)
                    emotion_categories = [categories[j] for j in range(len(probs))]
                    colors = [category_colors[j] for j in range(len(probs))]
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.bar(emotion_categories, probs, color=colors, edgecolor='gray')
                    ax.set_ylim(0, 1)
                    ax.set_ylabel("Probability", fontsize=12)
                    ax.set_xlabel("Emotion", fontsize=12)
                    ax.set_title("Predicted Probabilities", fontsize=12)

                    for idx, val in enumerate(probs):
                       ax.text(idx, val + 0.02, f"{val:.2f}", ha='center', fontsize=10)
                   
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    st.pyplot(fig)
                

            elif model_choice == "Classical ML (Ensemble Learning (LogRegr, SVM, NB))":
                st.success("✅ Using Classical ML model for emotion detection...")

                with st.spinner("Analyzing your data"):
                    time.sleep(2)
                    preprocessed_texts = preprocess_new_texts([user_text])
                    probs = model.predict_proba(preprocessed_texts)
                    pred_classes = model.predict(preprocessed_texts)
                    categories = {0:"sadness", 1:"joy", 2:"love", 3:"anger", 4:"fear", 5:"surprise"}

                    for i, text in enumerate([user_text]):
                        st.write("Original:", text)
                        st.write("Preprocessed:", preprocessed_texts[i])
                        predicted_emotion = categories[pred_classes[i]]
                        color = category_colors[pred_classes[i]]  # use the integer key to get color
                        st.markdown(f"<p style='font-size:18px; font-weight:bold; color:white; " f"background-color:{color}; padding:6px; border-radius:5px;'>"  f"🎯 Predicted Emotion: {predicted_emotion}</p>",unsafe_allow_html=True
                        )                     
                        prob_text = ", ".join([f"{categories[j]}:{probs[i][j]:.2f}" for j in range(len(probs[i]))])
                        st.write("Probabilities:", prob_text)
                    # Plot probabilities
                    prob_values = probs[0]  
                    emotion_categories = [categories[j] for j in range(len(prob_values))]
                    colors = [category_colors[j] for j in range(len(prob_values))]

                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.bar(emotion_categories, prob_values, color=colors, edgecolor='gray')
                    ax.set_ylim(0, 1)
                    ax.set_ylabel("Probability", fontsize=12)
                    ax.set_xlabel("Emotion", fontsize=12)
                    ax.set_title("Predicted Probabilities", fontsize=12)
  
                    # Remove top and right spines
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)



                    # Add probability labels on top of bars
                    for idx, val in enumerate(prob_values):
                        ax.text(idx, val + 0.02, f"{val:.2f}", ha='center', fontsize=10)
                    
                    st.pyplot(fig)

   
   # -------------------- IMAGE ANALYSIS --------------------
if st.session_state.page == "Image Analysis":
    st.header("🖼 Image Emotion Analysis")
    st.write("Upload a clear face photo (jpg/png). The CNN model will predict the emotion.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open image directly from upload
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=False)  # show original size

        # Save if needed for cv2
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image.save(temp_file, format="JPEG")
        temp_path = temp_file.name

        # Prediction
        with st.spinner("Analyzing image..."):
            time.sleep(2)
            pred_label, probs = predict_emotion(cnn_model, temp_path)

        if pred_label:
            st.subheader(f"🎯 Predicted Emotion: **{pred_label}**")

            # Show raw probabilities
            st.write("### Probability Distribution:")
            prob_text = ", ".join([f"{inv_label_map[i]}: {probs[i]:.2f}" for i in range(len(probs))])
            st.write(prob_text)

            # Plot probabilities
            emotion_categories = [inv_label_map[i] for i in range(len(probs))]
            colors = plt.cm.Paired(np.linspace(0, 1, len(probs)))

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(emotion_categories, probs, color=colors, edgecolor='gray')
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability", fontsize=12)
            ax.set_xlabel("Emotion", fontsize=12)
            ax.set_title("Predicted Probabilities", fontsize=14)

            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Add probability labels on top of bars
            for idx, val in enumerate(probs):
                ax.text(idx, val + 0.02, f"{val:.2f}", ha='center', fontsize=10)

            st.pyplot(fig)
