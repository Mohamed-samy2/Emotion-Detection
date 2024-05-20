import numpy as np
import tensorflow as tf
from tensorflow import keras
import streamlit as st
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import streamlit.components.v1 as components
from streamlit import components
from PIL import Image
from roberta_prediction import roberta
from statistics import mode
from streamlit_modal import Modal



title = st.title("Emotion Detector")

text = st.text_area("Express Your Feeling","",height=200)

with st.sidebar:
    # Use CSS to center the button
    st.markdown(
        """
        <style>
        .stButton>button {
            margin: 0 auto;
            display: block;
            # width: 170px; /* Set the width of the button */
            # height: 50px; /* Set the height of the button */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
def predict():
    roberta_model=roberta(text)
    prediction=roberta_model.predict()
    return prediction





show_popup = st.button(label='Submit')

if show_popup :
    # Call the predict function to obtain the predicted class
    pred = predict()

    if pred=='fear':
        pred = "Fear" + '😱'

    elif pred=='joy':
        pred= 'Joy' + '😄'

    elif pred=='sadness':
        pred='Sad' + '😔'

    elif pred=='anger':
        pred='Angry'+'😡'

    elif pred=='surprise':

        pred= 'Surprised'+'😯'
    else:
        pred ='Love'+'🥰'



    # Create a modal with the predicted class as title
    modal = Modal(
        key="demo-modal",
        title=pred,
        # Optional
        padding=60,  # default value
        max_width=500  # default value
    )
    # Define the content of the modal
    with modal.container():
        # Display the content of the modal
        st.write(" ")

        # Add a button at the bottom right to close the modal
        if st.button("OK", key="close-modal"):
            modal.toggle()





