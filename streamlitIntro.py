# import streamlit as st
# import pandas as pd
# import numpy as np
# import math
# import random

# st.write('Hi KUSHAL!')
import tensorflow as tf
import streamlit as st
@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('./second(1).hdf5')
    return model