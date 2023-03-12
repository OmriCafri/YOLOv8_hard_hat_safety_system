import datetime as dt
import os
import sys

import cv2
import pandas as pd

# Handling relative imports errors
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from firebase_manager import crud_manager


# Bounding boxes constants
# Colors dictionary
COLORS = {0: (0, 0, 255), 1: (0, 255, 0)}
# Texts dictionary
TEXTS = {0: 'Violation', 1: 'Hard Hat'}


def _add_bounding_boxes(prediction):
    """
    Function for adding bounding boxes to images
    (based on the prediction)

    Parameters
    ----------
    prediction: dict
        Prediction dictionary

    Returns
    -------
    numpy.ndarray
        An image with bounding boxes
    """
    img = prediction.orig_img
    for box, cls in zip(prediction.boxes.xyxy.cpu(),
                        prediction.boxes.cls.cpu()):
        # Converting tensor to a list of 4 numbers
        box = [int(coord) for coord in box]
        # Adding bounding boxes
        img = cv2.rectangle(img,
                            (box[0], box[1]),
                            (box[2], box[3]),
                            color=COLORS[int(cls)],
                            thickness=2)
        # Adding text
        img = cv2.putText(img,
                          TEXTS[int(cls)],
                          (box[0], box[1] - 10),
                          fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                          fontScale=0.5,
                          color=COLORS[int(cls)],
                          thickness=1,
                          lineType=cv2.LINE_AA)
    return img


def on_predict_postprocess_end(predictor):
    """
    A callback function for adding timestamp and bounding boxes to the image of the prediction results
    """
    for result in predictor.results:
        # Adding timestamp
        result.timestamp = dt.datetime.now()
        # Adding bounding boxes to the original image
        result.orig_img = _add_bounding_boxes(result)


def save(predictions, violation_images_dir=None):
    """
    Function for parsing prediction results,
    saving to firebase,
    and saving violation images if needed.

    Parameters
    ----------
    predictions: list[dic,...]
        List of predictions dictionary
    violation_images_dir: str (optional)
        Path for saving images of violations
        ** the path should be directory path and should end with / **
    """
    # Results to stream
    docs = []
    for _ in range(len(predictions)):
        pred = predictions.pop(0)
        # Parsing results
        docs.append({'people_detected': len(pred.boxes.cls),
                     'violations': int((pd.Series(pred.boxes.cls.cpu()) == 0).sum()),
                     'timestamp': str(pred.timestamp)})

        # Saving violations locally
        if violation_images_dir is not None and docs[-1]['violations'] > 0:
            cv2.imwrite(violation_images_dir + str(pred.timestamp).replace(':', '_') + '.png', pred.orig_img)

    # Docs names
    names = [str(doc['timestamp']) for doc in docs]

    # Writing to firebase
    crud_manager.create(documents=docs, documents_names=names)
