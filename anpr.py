from PIL import Image
import streamlit as st
from hydralit import HydraHeadApp

from detect import main_runner


class ANPRApp(HydraHeadApp):
    def run(self):
        st.title("Licence Plate detection system")
        uploaded_file_path = st.file_uploader("Choose an image...", type=["jpg", "png"])
        image = Image.open(uploaded_file_path)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        main_runner(uploaded_file_path)
        st.write("")
        st.write("Classifying...")
