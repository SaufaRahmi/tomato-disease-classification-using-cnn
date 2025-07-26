import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the model once
@st.cache_resource
def load_cnn_model():
    model = load_model('model.h5')  # change this to your model filename
    return model

model = load_cnn_model()

# Set title and instructions
st.title("🍅 Tomato Disease Classifier")
st.write("""
Upload a tomato leaf image, and this app will predict the disease type using a pre-trained CNN model.
\n**Instructions:**
- Upload a clear image of a tomato leaf.
- Wait for the prediction.
""")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    st.write("Processing image and making prediction...")
    
    img = img.resize((256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)

    # Show result
    predicted_class = np.argmax(prediction, axis=1)[0]

    # You can define class names like this:
    class_names = ['Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold', 'Septoria Leaf Spot', 
                   'Spider Mites Two Spotted Spider Mite', 'Target Spot', 'Yellow Leaf Curl Virus', 'Mosaic Virus',
                   'Healthy Tomato']
    st.success(f"✅ Prediction: **{class_names[predicted_class]}**")
