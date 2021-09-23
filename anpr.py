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
        image.save(f"/content/yolov5custom/{uploaded_file_path.name}")
        if image is not None:
            # run_cleanup()
            st.image(image, caption="Uploaded Image", use_column_width=True)
            saved, crop = path_builder(uploaded_file_path.name)
            # st.text(f"{saved},{crop},{uploaded_file_path.name}")
            if st.button("Find License plate"):
                with st.spinner("Loading Model for detecting images"):
                    main_runner(uploaded_file_path.name)

                st.success("Finished and saved predictions")
                st.text("### Detected licenses")

                saved = Image.open(saved)
                st.image(saved, caption="Image with license plates", use_column_width=True)

                # if st.button("Gather text"):
                with st.spinner("Loading OCR"):
                    predicted_groups, fig = make__predictions(crop,pipeline)
                st.pyplot(fig)
                chunks = f""
                for predicted_group in predicted_groups:
                    for chunk in predicted_group:
                      chunks+=f"{chunk[0]} "
                        # st.text(chunk[0])
                st.text(chunks)
                # st.text(predicted_groups)
                # st.table(predicted_groups)
