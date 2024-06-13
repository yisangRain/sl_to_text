# COSC428 - multiple bounding box sign language detection


## Libraries used
- OpenCV
- MediaPipe (Holistic Solution)
- Click
- SQLite
- OS
- CSV
- Time

## Introduction
This software has two features: learning and reading.

### Learn mode
Learn mode converts, from image or video source, the given human torso body pose and hand joint positions into a set of strings, one per torse and hand. This set of string is then recorded into the SQLite database with the user-assigned label.

Batch learning is available for both video and directory sources.

### Read mode
The read mode also converts the human torso and hand positions into a set of strings. These strings are then used to query the database to return a label if there is a match. The resulting label is then displayed as an overlay to the original video frame.

## How to use

This software has full CLI support

To start, on cmd window run the main.py with selection of CLI parameters:

- --init default=read, 'learn'
    default option 'read'
    Initiate the program in the given mode.
- --source <source file path>
    Path to the source file. Accepts 'cam' for live camera stream input. mp4, jpg, png formats also accepted.
- --fresh default=False
    Reset database
- --batch default=False
    Batch learning from a given directory path in the source or videofile
- --analysis default=False
    Runs the program in headless mode and generates analysis report of the detected label per frame. Assigns None if no sign has been detected for the frame.

### Read from video:
```
python3 main.py --source path/to/the/video.mp4
```

### Read from cam:
```
python3 main.py --source cam
``` 

### Learn from a single image:
```
python3 main.py --init learn --source path/to/the/image.jpg

Label? 
<enter label>
```

### Batch learn from a directory containing images only
```
python3 main.py --init learn --source path/to/the/imagefolder

Batch learning. Label? 
<Enter label>
``` 

### Batch learn from a video
```
python3 main.py --init learn --source path/to/the/video.mp4

Batch learning. Label? 
<Enter label>
```

For entering label, conclude with the enter key

