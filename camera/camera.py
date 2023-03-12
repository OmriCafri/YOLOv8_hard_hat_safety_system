import os
import sys
import time
import datetime as dt

import ultralytics
import keyboard
import threading
import yaml

# Handling relative imports errors
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from _camera_utils import save, on_predict_postprocess_end

try:
    with open('../configurations.yaml', 'r') as f:
        conf = list(yaml.load_all(f, Loader=yaml.SafeLoader))[0]
except:
    with open('configurations.yaml', 'r') as f:
        conf = list(yaml.load_all(f, Loader=yaml.SafeLoader))[0]

# Reading important paths
path = conf['production_model']
# Path for saving violation images
saving_dir = conf['violations_path']

# Quitting flag
EXIT_PROGRAM = False
# Inference results will be saved to this list
predictions = []


def _quit(violation_images_dir):
    """
    Function for finishing execution when hot key is pressed ('esc')

    Parameters
    ----------
    violation_images_dir: str
        Path for saving images of violations
        ** the path should be directory path and end with / **

    Returns
    -------

    """
    global EXIT_PROGRAM
    EXIT_PROGRAM = True
    print('Program Execution Will End After The Last Iteration\n** Please Wait **')
    # executing the last stream to firebase
    save(predictions, violation_images_dir)


def _save_periodically(buffer=60, violation_images_dir=None):
    """
    Function for streaming results to firebase.

    Parameters
    ----------
    buffer: int (default 60)
        Seconds between each streaming iteration
    violation_images_dir: str (optional)
        Path for saving images of violations
        ** the path should be directory path and end with / **
    """
    global predictions
    last_streaming = dt.datetime.now()
    # Until 'esc' keyboard is pressed
    while not EXIT_PROGRAM:
        # Considering buffer
        if (dt.datetime.now() - last_streaming).seconds >= buffer:
            # Updating last prediction timestamp
            last_streaming = dt.datetime.now()
            # streaming to firebase
            save(predictions, violation_images_dir)
        time.sleep(0.5)


def _run_camera_prediction(buffer=5, show=False):
    """
    Function for getting predictions for live camera footage.
    (The results will be appended to "predictions" global variable)

    Parameters
    ----------
    buffer: int (default 5)
        Seconds between each prediction
    show: bool (default True)
        Whether to show a screen with predictions or not
    """
    global predictions
    # Creating model instance from pretrained weights
    model = ultralytics.YOLO(path)
    # Adding a callback
    model.add_callback("on_predict_postprocess_end", on_predict_postprocess_end)
    # Creating stream for inference
    results = model.predict(source=0, show=show, stream=True)
    # Initiating variable for saving the last prediction time
    last_prediction = dt.datetime.now()
    # Until 'esc' keyboard is pressed
    while not EXIT_PROGRAM:
        # Considering buffer
        if (dt.datetime.now() - last_prediction).seconds >= buffer:
            # Updating last prediction timestamp
            last_prediction = dt.datetime.now()
            # Inference
            predictions.append(next(results))
        time.sleep(0.5)


def run(prediction_buffer=5, saving_buffer=60, violation_images_dir=None, show=False):
    """
    Function for running "Camera Prediction System".

    1. Connects to the local camera.
    2. Uses YOLOv8 model to detect hard hat violations.
    3. Streams results to firebase
    4. Saves violation images locally

    Parameters
    ----------
    prediction_buffer: int (default 5)
        Buffer between two sequential predictions, in seconds.
        Must be higher than 0.5 (firebase limitation - can be changed)
    saving_buffer: int (default 60)
        Buffer between two sequential saves to firebase, in seconds.
        Must be higher than 0.5 (firebase limitation - can be changed)
    violation_images_dir: str (optional)
        Path for saving images of violations
        ** the path should be directory path and end with / **:
    show: bool (default False)
        Whether to show prediction images live or not.
        If True, a popup screen will appear.
    """
    # Checking if violation_images_dir exists (if not None)
    if violation_images_dir is not None and not os.path.exists(violation_images_dir):
        raise Exception('Directory for saving images does not exist')
    # Setting hotkey for quitting the program
    keyboard.add_hotkey('esc', lambda: _quit(violation_images_dir))
    # Creating a thread for streaming data to firebase
    threading.Thread(target=lambda: _save_periodically(buffer=saving_buffer,
                                                       violation_images_dir=violation_images_dir)).start()
    # Inference
    _run_camera_prediction(buffer=prediction_buffer,
                           show=show)


if __name__ == '__main__':
    run(prediction_buffer=1, saving_buffer=15, violation_images_dir=saving_dir, show=True)