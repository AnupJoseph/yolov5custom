import keras_ocr
import matplotlib.pyplot as plt
def make_pipeline():
    return keras_ocr.pipeline.Pipeline()

def make__predictions(image_path,pipeline):
    image = keras_ocr.tools.read(image_path)
    prediction_groups = pipeline.recognize([image])
    fig= plt.figure( figsize=(20, 20))
    ax = keras_ocr.tools.drawAnnotations(image=image, predictions=prediction_groups[0])
    return prediction_groups,fig,
