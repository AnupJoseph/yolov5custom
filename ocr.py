import keras_ocr
import matplotlib.pyplot as plt
import cv2

def make_pipeline():
    return keras_ocr.pipeline.Pipeline()

def make__predictions(image_path,pipeline):
    image = cv2.imread(image_path)
    scale_per = 3
    width = int(image.shape[1] * scale_per)
    height = int(image.shape[0] * scale_per)
    dim = (width, height)
    scaled = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
    des = cv2.fastNlMeansDenoisingColored(scaled, None, 10, 10, 7, 15)
    cv2.imwrite(image_path, des)
    image = keras_ocr.tools.read(image_path)
    prediction_groups = pipeline.recognize([image])
    fig,ax= plt.subplots( nrows=1,figsize=(20, 20))
    ax = keras_ocr.tools.drawAnnotations(image=image, predictions=prediction_groups[0],ax=ax)
    return prediction_groups,fig
