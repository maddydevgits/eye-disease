import streamlit as st
from PIL import Image

from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

def process_result():
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open("uploads/test.jpg").convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    st.success("Class:" + str(class_name))
    st.success("Confidence Score:" + str(confidence_score))

def open_image(img):
    return(Image.open(img))

st.title('Eye Disease Detector')

img_file=st.file_uploader('Upload your Eye Image',type=['png','jpg','jpeg'])

if img_file is not None:
    file_details={}
    file_details['type']=img_file.type
    file_details['size']=img_file.size
    file_details['name']=img_file.name
    st.write(file_details)

    st.image(open_image(img_file),width=250)

    with open('uploads/test.jpg','wb') as f:
        f.write(img_file.getbuffer())
    
    process_result()