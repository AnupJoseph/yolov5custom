# Externals
from PIL import Image
import streamlit as st
from hydralit import HydraHeadApp


# Project specific
from detect import main_runner
from ocr import make_pipeline, make__predictions

# Standard library
import pathlib

pipeline = make_pipeline()


def path_builder(path):
    path = pathlib.Path(path)
    name = path.name
    saved_path = (
        pathlib.Path("/content/yolov5custom/runs/detect/yolo_plate_detection") / name
    )
    crop_path = (
        pathlib.Path(
            "/content/yolov5custom/runs/detect/yolo_plate_detection/crops/plate"
        )
        / name
    )
    return str(saved_path), str(crop_path)


class ANPRApp(HydraHeadApp):
    def run(self):
        st.title("Licence Plate detection system")
        uploaded_file_path = st.file_uploader("Choose an image...", type=["jpg", "png"])
        image = Image.open(uploaded_file_path)
        if image is not None:
            # run_cleanup()
            st.image(image, caption="Uploaded Image", use_column_width=True)
            saved, crop = path_builder(uploaded_file_path)
            if st.button("Find License plate"):
                with st.spinner("Loading Model for detecting images"):
                    main_runner(uploaded_file_path)

                st.success("Finished and saved predictions")
                st.text("### Detected licenses")

                saved = Image.open(saved)
                st.image(saved, caption="Uploaded Image", use_column_width=True)

                if st.button("Gather text"):
                    predicted_groups, fig = make__predictions(crop,pipeline)
                    st.pyplot(fig)
                    st.text(predicted_groups)
                    # st.table(predicted_groups)
