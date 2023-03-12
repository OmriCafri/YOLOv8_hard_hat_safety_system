import sys

import yaml
import ultralytics
import pandas as pd
import cv2

try:
    with open('../configurations.yaml', 'r') as f:
        conf = list(yaml.load_all(f, Loader=yaml.SafeLoader))[0]
except:
    with open('configurations.yaml', 'r') as f:
        conf = list(yaml.load_all(f, Loader=yaml.SafeLoader))[0]

# path for production model
path = conf['production_model']


def _video_duration(video_path):
    """
    Function for getting video length in seconds

    Parameters
    ----------
    video_path: str
        Path for video

    Returns
    -------
    float
        Video duration in seconds
    """
    # create video capture object
    data = cv2.VideoCapture(video_path)

    # count the number of frames
    frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = data.get(cv2.CAP_PROP_FPS)

    # calculate duration of the video
    return round(frames / fps)


def _parse_results(results, video_path):
    """
    Function for parsing results

    Parameters
    ----------
    results: list[dict,...]
        Inference results
    video_path: str
        Path for video
    Returns
    -------
    pd.DataFrame
        A report with the following columns:
        timestamp, people_detected, violations
    """
    # Getting video duration
    video_duration = _video_duration(video_path)
    # Parsing results
    parsed = {'timestamp': [], 'people_detected': [], 'violations': []}

    for ind, result in enumerate(results):
        parsed['timestamp'].append(str(round(video_duration * (ind / len(results)) / 3600)) + ' : ' +
                                   str(round(video_duration * (ind / len(results)) / 60)) + ' : ' +
                                   str(round(video_duration * (ind / len(results))) % 60))
        parsed['people_detected'].append(len(result.boxes.cls))
        parsed['violations'].append(int((pd.Series(result.boxes.cls.cpu()) == 0).sum()))
    return pd.DataFrame(parsed)


def run(video_path, show=False, save_video=True, report_path=None):
    """
    Function for running model on mp4 videos.

    Parameters
    ----------
    video_path: str
        Path for the video we would like to predict on
    show: bool (default False)
        Whether to show the predictions live or not
    save_video: bool (default True)
        Whether to save the product video (with bounding boxes)
    report_path: str (optional)
        Path for saving an excel file report with model results.
        (timestamp - people detected - violations)
    """
    # Loading model instance
    model = ultralytics.YOLO(path)
    # Run the model on video
    results = model.predict(video_path, save=save_video, stream=False, show=show)
    # Parsing results and saving to excel
    if report_path is not None:
        report = _parse_results(results, video_path)
        report.to_excel(report_path, index=False)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        run(video_path=sys.argv[1])
    elif len(sys.argv) == 3:
        run(video_path=sys.argv[1], report_path=sys.argv[2])
    else:
        raise Exception('Too many / few args please use python interface for more options')