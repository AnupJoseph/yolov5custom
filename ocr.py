import keras_ocr
def make_pipeline():
    return keras_ocr.pipeline.Pipeline()

def make__predictions(image_path,pipeline):
    image = keras_ocr.tools.read(image_path)
    prediction_groups = pipeline.recognize([image])
