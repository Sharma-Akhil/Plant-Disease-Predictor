import streamlit as st
import tensorflow as tf


@st.cache_resource
def load_model():
  model=tf.keras.models.load_model(r'/Users/akhilsharma/potato/my_model2.hdf5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()
st.write("""
         # Potato plant Disease Prediction
         """
         )

file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    
        size = (256,256)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    st.write(predictions)
    st.write(score)
    class_names=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
    st.write("""
    # This image most likely belongs to {} with a {:.2f} percent confidence.
    """
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
