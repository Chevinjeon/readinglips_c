# Import all the dependencies
import streamlit as st
import os 
import imageio
import sys

import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model
from tensorflow import keras


# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Set up the sidebar
with st.sidebar: 
    st.image('https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.pngwing.com%2Fen%2Ffree-png-zddrs&psig=AOvVaw3-0jBjHM-yA7Hl1EqMXN37&ust=1683445615819000&source=images&cd=vfe&ved=0CBEQjRxqFwoTCNDIuOGZ4P4CFQAAAAAdAAAAABAI')
    st.title('Lip Reading')
    st.info('Trained with the LipNet deep learning model.')


st.title('LipBud Web App') 
# Generating a list of options or videos 
options = os.listdir(os.path.join('.', 'data', 's1'))
#print(options)


selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options: 

    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('data','s1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')
        # Rendering the app
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)

    with col2: 
        st.info('This is all the machine learning model sees when making a prediction')
        print("bbbbbb", file_path)
        string_tensor_filepath = tf.constant(selected_video)
        video, annotations = load_data(tf.convert_to_tensor(string_tensor_filepath))
        imageio.mimsave('animation.gif', video, fps=10)
        st.image('animation.gif', width=400) 

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

# convert prediction to text 
        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
