# YOLOv8 - Hard Hat Safety System

The project has two main components:

1. "Camera" - Connects to the local webcam, detects violations in real time and streams the results to a firebase DB.
2. "Video" - Detects and reports violation in a mp4 file.

![alt text](https://github.com/OmriCafri/YOLOv8_hard_hat_safety_system/blob/master/inference_results.png?raw=true)
## Installation

## Installation

First download the project to your local machine.

Run the following command for installing dependencies.

```bash
pip install -r requirements
```

You might need to install cudatoolkit (For inference on GPU ).

This can be done using conda.
```bash
conda install cudatoolkit
```
Get your firebase key and paste it in **firebase_manager/creds/**

Feel free to follow the instructions in the following tutorial:
[firebase using python](https://www.analyticsvidhya.com/blog/2022/07/introduction-to-google-firebase-firestore-using-python/)


Now open **configuration.yaml** and change the specified paths to paths matching your machine.
## Usage

Starting the program can be done either with python or command prompt interface.

**Python**
```python
# Importing the camera component
from camera import camera

# Running the model - camera
camera.run(prediction_buffer=1, 
           saving_buffer=15,
           violation_images_dir=saving_dir,
           show=True) # see documentation for further explanation


# Importing the video component
from video import video

# Running the model - video
video.run(video_path='site1.mp4',
          report_path='report.xlsx')
```

**Command Prompt**

Running the camera component
```bash
python camera/camera.py
```

Running the video component

python video/video.py <video_path> <path_for_saving_report>

```bash
 python video/video.py D:\projects\hard_hat_safety_system_YOLOv8\video\site1.mp4 D:\projects\hard_hat_safety_system_YOLOv8\video\report2.xlsx
```
## Contact

Feel free to contact me:

**omricafri100@gmail.com**