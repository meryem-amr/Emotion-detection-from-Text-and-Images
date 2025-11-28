# Emotion-detection-from-Text-and-Images
Emotion Detection from Text and Images is an AI application that identifies emotions from user-provided text or images. It uses neural network models for classification and is built with Python, featuring a Streamlit interface and a Python-based backend for processing and prediction.
Emotion Detection from Text and Images

This project is an AI-based application that detects human emotions from text or images. It uses deep learning models to classify emotions and provides an easy-to-use interface built with Streamlit.

- **Features**

1. Text Emotion Detection using NLP neural network models
2. Image Emotion Detection using CNN-based models
3. Real-time classification
4. Simple and interactive Streamlit UI
5. Python-based backend for preprocessing and prediction

- **Technologies Used**

1. Python
2. Streamlit (UI)
3. TensorFlow (Deep Learning)
4. OpenCV / PIL (Image processing)
5. NumPy & Pandas

- **Project Structure**
├── models/
│   ├── text_model.h5
│   └── image_model.h5
├── app.py
├── utils/
│   ├── preprocess_text.py
│   ├── preprocess_image.py
│   └── predict.py
├── requirements.txt
└── README.md

- **Installation**

1. Clone the repository:
git clone <your-repo-link>
cd emotion-detection

2. Install dependencies:
pip install -r requirements.txt


3. Run the application:
streamlit run app.py

- **How It Works**

1. Choose Text or Image input
2. The system preprocesses the input
3. Neural network model predicts the emotion
4. Results are displayed instantly in the UI

- **Supported Emotions**

1. Happy
2. Sad
3. Angry
4. Fear
5. Surprise
6. Neutral

- **Author** 
- Meryem Amr (feel free to update this section)
