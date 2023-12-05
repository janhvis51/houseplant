import streamlit as st
import pandas as pd
import numpy as np
import math
import random
import requests
import tensorflow as tf
import streamlit as st
st.title("Image Classification")
@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('./second1.hdf5')
    return model
#Title

classes=["UnHealthy Snake Plant","UnHealthy Syngonium podophyllum","UnHealthy Adonidia merrillii","Healthy Syngonium podophyllum","Healthy Snake Plant","Healthy Adonidia merrillii"]

# image preprocessing
def load_image(image):
    img=tf.image.decode_png(image,channels=3)
    img=tf.cast(img,tf.float32)
    img/=255.0
    img=tf.image.resize(img,(256,256))
    img=tf.expand_dims(img,axis=0)
    return img

#Get image URL from user
image_path=st.text_input("Enter Image URL to classify...")

#Get image from URL and predict
if image_path:
    try:
        content=requests.get(image_path).content
        st.write("Predicting Class...")
        with st.spinner("Classifying..."):
            img_tensor=load_image(content)
            model = load_model()
            pred=model.predict(img_tensor)
            pred_class=classes[np.argmax(pred)]
            st.write("Predicted Class:",pred_class)
            st.image(content,use_column_width=True)
    except:
        st.write("Invalid URL")
    # try:
    #     content = requests.get(image_path).content
    #     st.write("Predicting Class...")
    #     with st.spinner("Classifying..."):
    #         img_tensor = load_image(content)
    #         model = load_model()
    #         pred_class = model.predict(img_tensor)
    #         st.write("Predicted Class:", pred_class)
    #         st.image(content, use_column_width=True)
    # except Exception as e:
    #     st.write(f"Error: {e}")