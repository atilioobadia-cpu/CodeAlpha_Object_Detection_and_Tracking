# CodeAlpha Object Detection and Tracking

[![CI](https://github.com/atilioobadia-cpu/CodeAlpha_Object_Detection_and_Tracking/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/atilioobadia-cpu/CodeAlpha_Object_Detection_and_Tracking/actions/workflows/ci.yml)

A local web application for detecting and tracking objects in webcam streams or uploaded video. OpenCV reads and prepares each frame, a pretrained YOLO26n model detects objects, and ByteTrack maintains object IDs across consecutive frames.

## Features

- Live webcam capture through OpenCV.
- MP4, MOV, AVI, MKV, and WEBM video upload (up to 150 MB).
- Pretrained YOLO object detection with adjustable confidence.
- ByteTrack object IDs drawn alongside class labels and bounding boxes.
- Live object counts, processing speed, and analyzed-frame metrics.
- Local processing; no video is sent to a hosted service.

## Requirements

- Python 3.10 or newer
- A webcam for live capture, or a local video file
- Internet access on first run to download the pretrained model weights

## Run locally

From this directory, create a virtual environment, install dependencies, and launch the Flask server:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python server.py
```

Open [http://127.0.0.1:4176](http://127.0.0.1:4176). On the first tracking session, Ultralytics downloads `yolo26n.pt` and caches the weights for later runs. The model can be changed with the `VISION_MODEL` environment variable. The server port can be changed with `PORT`.

## Use the tracker

1. Choose **Webcam** or **Video file**.
2. For a file, select a supported video from your computer.
3. Adjust detection confidence if needed.
4. Select **Start tracking**. The first run may take longer while the pretrained model downloads.
5. Follow the labeled boxes and track IDs in the preview. Use **Stop session** to end processing.

## How it works

OpenCV `VideoCapture` reads a camera or uploaded video frame by frame. The pretrained YOLO model detects objects in each frame. Ultralytics tracking mode runs ByteTrack with persistent state so IDs are associated across sequential frames. OpenCV encodes the annotated frames as an MJPEG stream for the web preview.

## Privacy and limits

Video files are handled locally by the Flask process and temporary uploads are removed when processing ends. Webcam access is provided by OpenCV on the computer running the server. CPU-only systems may process fewer frames per second; use a shorter or lower-resolution video for smoother results.

## License and references

The application source is provided for the CodeAlpha internship project. The pretrained model and Ultralytics package are subject to their own licenses and terms. See [Ultralytics tracking documentation](https://docs.ultralytics.com/modes/track/) and [OpenCV video capture documentation](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html) for the upstream APIs used.
