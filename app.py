import streamlit as st
import numpy as np
from PIL import Image
import keras
from keras.preprocessing.image import img_to_array

# Load your trained model
model = keras.models.load_model('./melanoma_model.h5')

# Function to preprocess the image so it can be input into the model
def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0
    return image

# Set up the Streamlit app
st.title("Mole Classification App")
st.write("Upload an image of a mole to classify it as benign or malignant.")

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)
    class_names = ['Benign', 'Malignant']  # You may need to change these based on your model's classes
    string = "This mole is most likely: " + class_names[np.argmax(prediction)]
    
    st.success(string)


# # Streamlit app
# st.title("Melanoma Detector")

# # Upload an image
# uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])

# if uploaded_image is not None:
#     # Display the uploaded image
#     st.image(uploaded_image, caption='Uploaded Image.', use_column_width=True)

#     # Perform classification when a user uploads an image
#     if st.button('Classify'):
#         # Load the model
#         model = load_model('./melanoma_model.h5')

#         # Preprocess the uploaded image
#         processed_image = preprocess_image(uploaded_image)

#         # Perform image classification using the model
#         result = predict_class(processed_image, model)

#         # Display the classification result
#         st.write(f"Prediction: {result}")
