# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image

# # Load Rice model
# @st.cache_resource
# def load_rice_model():
#     try:
#         return tf.keras.models.load_model(r"C:\Users\abc\Desktop\Final Project\Plant_Disease_Prediction\rice_model\trained_plant_disease_model.keras")
#     except Exception as e:
#         st.error(f"Error loading Rice Model: {str(e)}")

# # Load Sugarcane model
# @st.cache_resource
# def load_sugarcane_model():
#     try:
#         return tf.keras.models.load_model(r"C:\Users\abc\Desktop\Final Project\Plant_Disease_Prediction\sugarcane_model\trained_plant_disease_model.keras")
#     except Exception as e:
#         st.error(f"Error loading Sugarcane Model: {str(e)}")

# rice_model = load_rice_model()
# sugarcane_model = load_sugarcane_model()

# # Prediction Function
# def predict_disease(model, image):
#     try:
#         img = Image.open(image)
#         img = img.resize((128, 128))
#         img_array = np.array(img) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)
#         prediction = model.predict(img_array)
#         return np.argmax(prediction), np.max(prediction)  # Return index and confidence
#     except Exception as e:
#         st.error(f"Error during prediction: {str(e)}")
#         return None, None

# # Disease Classes
# rice_classes = (
#     "Bacterial Leaf Blight", "Brown Spot", "Healthy", "Leaf Blast", "Narrow Brown Spot"
# )

# sugarcane_classes = (
#     "Leaf Scald", "Mosaic", "Red Rot", "Rust", "Yellow Disease"
# )

# # Accuracy (optional)
# rice_accuracy = 94.5
# sugarcane_accuracy = 92.3

# # Sidebar
# st.sidebar.title("Dashboard")
# app_mode = st.sidebar.selectbox("Select Page", [
#     "Home", "About", "Rice Disease Recognition", "Sugarcane Disease Recognition"
# ])

# # Home Page
# if app_mode == "Home":
#     st.header("RICE & SUGARCANE DISEASE RECOGNITION SYSTEM")
#     st.image(r"C:\Users\abc\Desktop\Final Project\Plant_Disease_Prediction\home.jpg", use_column_width=True)
#     st.markdown("""
#     Welcome to the **Rice & Sugarcane Disease Recognition System**! 🌿🦠  
#     Upload an image of a leaf and detect common diseases using deep learning models.  
#     Select a crop from the sidebar to get started.
#     """)

# # About Page
# elif app_mode == "About":
#     st.header("About")
#     st.markdown("""
#     #### Dataset Info:
#     - Rice Diseases: Bacterial Leaf Blight, Brown Spot, Leaf Blast, etc.
#     - Sugarcane Diseases: Leaf Scald, Mosaic, Red Rot, etc.

#     #### Model Accuracies:
#     - Rice Model: **94.5%**
#     - Sugarcane Model: **92.3%**

#     Trained using CNN-based architecture for precise classification.
#     """)

# # Rice Disease Prediction
# elif app_mode == "Rice Disease Recognition":
#     st.header("Rice Disease Recognition 🌾")
#     uploaded_file = st.file_uploader("Upload a Rice Leaf Image", type=["jpg", "jpeg", "png"])
    
#     if uploaded_file:
#         st.image(uploaded_file, use_column_width=True)
#         if st.button("Predict Rice Disease"):
#             st.write("Analyzing image...")
#             result, confidence = predict_disease(rice_model, uploaded_file)
#             if result is not None:
#                 prediction = rice_classes[result]
#                 st.success(f"Predicted Rice Disease: **{prediction}**")
#                 st.info(f"Confidence: {confidence*100:.2f}%")
#                 st.info(f"Model Accuracy: {rice_accuracy}%")

# # Sugarcane Disease Prediction
# elif app_mode == "Sugarcane Disease Recognition":
#     st.header("Sugarcane Disease Recognition 🌿")
#     uploaded_file = st.file_uploader("Upload a Sugarcane Leaf Image", type=["jpg", "jpeg", "png"])
    
#     if uploaded_file:
#         st.image(uploaded_file, use_column_width=True)
#         if st.button("Predict Sugarcane Disease"):
#             st.write("Analyzing image...")
#             result, confidence = predict_disease(sugarcane_model, uploaded_file)
#             if result is not None:
#                 prediction = sugarcane_classes[result]
#                 st.success(f"Predicted Sugarcane Disease: **{prediction}**")
#                 st.info(f"Confidence: {confidence*100:.2f}%")
#                 st.info(f"Model Accuracy: {sugarcane_accuracy}%")

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load Rice model
@st.cache_resource
def load_rice_model():
    try:
        return tf.keras.models.load_model(
            r"C:\Users\abc\Desktop\Final Project\Plant_Disease_Prediction\rice_model\trained_plant_disease_model.keras"
        )
    except Exception as e:
        st.error(f"Error loading Rice Model: {str(e)}")

# Load Sugarcane model
@st.cache_resource
def load_sugarcane_model():
    try:
        return tf.keras.models.load_model(
            r"C:\Users\abc\Desktop\Final Project\Plant_Disease_Prediction\sugarcane_model\trained_plant_disease_model.keras"
        )
    except Exception as e:
        st.error(f"Error loading Sugarcane Model: {str(e)}")

rice_model = load_rice_model()
sugarcane_model = load_sugarcane_model()

# Prediction function
def predict_disease(model, image):
    try:
        img = Image.open(image).convert("RGB")  # Ensures 3-channel image
        img = img.resize((128, 128))            # Resize to match training size
        img_array = np.array(img) / 255.0       # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Batch dimension

        prediction = model.predict(img_array)

        st.write("Raw prediction output:", prediction)  # Debug: Remove in production

        predicted_class = np.argmax(prediction)
        confidence_score = np.max(prediction)

        return predicted_class, confidence_score
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

# Classes
rice_classes = ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy', 'Leaf Blast', 'Narrow Brown Spot']



sugarcane_classes = ['Healthy', 'Leaf Scald', 'Mosaic', 'Red Rot', 'Rust', 'Yellow Disease']

# Accuracy info
rice_accuracy = 94.5
sugarcane_accuracy = 92.3

# Sidebar Navigation
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", [
    "Home", "About", "Rice Disease Recognition", "Sugarcane Disease Recognition"
])

# Home Page
if app_mode == "Home":
    st.header("🌾 RICE & SUGARCANE DISEASE RECOGNITION SYSTEM 🌿")
    st.image(r"C:\Users\abc\Desktop\Final Project\Plant_Disease_Prediction\home.jpg", use_column_width=True)
    st.markdown("""
    Welcome to the **Rice & Sugarcane Disease Recognition System**!  
    Upload an image of a leaf to detect common diseases using deep learning models.  
    Select a crop from the sidebar to get started.
    """)

# About Page
elif app_mode == "About":
    st.header("📚 About")
    st.markdown("""
    #### Dataset Info:
    - **Rice Diseases:** Bacterial Leaf Blight, Brown Spot, Leaf Blast, etc.
    - **Sugarcane Diseases:** Leaf Scald, Mosaic, Red Rot, etc.

    #### Model Accuracies:
    - 🌾 Rice Model: **94.5%**
    - 🌿 Sugarcane Model: **92.3%**

    Both models are trained using Convolutional Neural Networks (CNNs) for high-accuracy classification.
    """)

# Rice Prediction
elif app_mode == "Rice Disease Recognition":
    st.header("🌾 Rice Disease Recognition")
    uploaded_file = st.file_uploader("Upload a Rice Leaf Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict Rice Disease"):
            st.write("Analyzing image...")
            result, confidence = predict_disease(rice_model, uploaded_file)

            if result is not None:
                prediction = rice_classes[result]
                st.success(f"Predicted Rice Disease: **{prediction}**")
                st.info(f"Confidence: **{confidence * 100:.2f}%**")
                st.info(f"Model Accuracy: **{rice_accuracy}%**")

# Sugarcane Prediction
elif app_mode == "Sugarcane Disease Recognition":
    st.header("🌿 Sugarcane Disease Recognition")
    uploaded_file = st.file_uploader("Upload a Sugarcane Leaf Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict Sugarcane Disease"):
            st.write("Analyzing image...")
            result, confidence = predict_disease(sugarcane_model, uploaded_file)

            if result is not None:
                prediction = sugarcane_classes[result]
                st.success(f"Predicted Sugarcane Disease: **{prediction}**")
                st.info(f"Confidence: **{confidence * 100:.2f}%**")
                st.info(f"Model Accuracy: **{sugarcane_accuracy}%**")




