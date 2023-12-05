# import streamlit as st
# import pandas as pd
# import numpy as np
# import math
# import random

import tensorflow as tf
import streamlit as st
st.title("Image Classification")
@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('./second1.hdf5')
    return model
#Title

classes=["UnHealthy Syngonium podophyllum","UnHealthy Snake Plant","UnHealthy Adonidia merrillii","Healthy Syngonium podophyllum","Healthy Snake Plant","Healthy Adonidia merrillii"]

# image preprocessing
def load_image(image):
    img=tf.image.decode_jpeg(image,channels=3)
    img=tf.cast(img,tf.float32)
    img/=255.0
    img=tf.image.resize(img,(28,28))
    img=tf.expand_dims(img,axis=0)
    return img

#Get image URL from user
image_path=st.text_input("Enter Image URL to classify...","snakeplant1.png")

#Get image from URL and predict
if image_path:
    try:
        content=requests.get(image_path).content
        st.write("Predicting Class...")
        with st.spinner("Classifying..."):
            img_tensor=load_image(content)
            pred=model.predict(img_tensor)
            pred_class=classes[np.argmax(pred)]
            st.write("Predicted Class:",pred_class)
            st.image(content,use_column_width=True)
    except:
        st.write("Invalid URL")