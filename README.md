# YOLO with Core ML

This repo was forked and modified from [hollance/YOLO-CoreML-MPSNNGraph](https://github.com/hollance/YOLO-CoreML-MPSNNGraph). Some changes I made:

1. Support mlmodel converted by coremltools 5.0.
2. Support any YOLO model converted by coremltools 5.0.
3. Support different input model size: 416, 512 or 608.


## About YOLO object detection

YOLO is an object detection network. It can detect multiple objects in an image and puts bounding boxes around these objects. [Read hollance's blog post about YOLO](http://machinethink.net/blog/object-detection-with-yolo/) to learn more about how it works.


To run the app:

1. Download the models from [here](https://drive.google.com/drive/folders/10RFA8cBVi33UF7OKiVVIV6Rb1mWdqtkI?usp=sharing), or download the weights file and cfg file, and convert it to mlmodel file by using my [YOLO CoreML converter](https://github.com/hwdavr/YOLO-CoreML-Converter)
- YOLOv3  
Weights file: https://pjreddie.com/media/files/yolov3.weights
Configure file: https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3.cfg  

- Tiny YOLOv3  
Weights file: https://pjreddie.com/media/files/yolov3-tiny.weights
Configure file: https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3-tiny.cfg  

- YOLOv4  
Weights file: https://drive.google.com/open?id=1bV4RyU_-PNB78G-OtoTmw1Q7t_q90GKY.  
If above link cannot work, please go to (YOLOv4-model-zoo)[https://github.com/AlexeyAB/darknet/wiki/YOLOv4-model-zoo] and download YOLOv4-Mish-416 or YOLOv4-Leaky-416 weights file.  
Configure file: https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg  

- Tiny YOLOv4: 
Weights file: https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights
If above link canot work, please go to https://github.com/AlexeyAB/darknet/releases/tag/yolov4 and download file yolov4-tiny.weights.  
Configure file: https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg

2. Copy CoreML model in Models folder
3. Open the **xcodeproj** file in Xcode and run it on a device with iOS 13 or above and install.

The reported "elapsed" time is how long it takes the YOLO neural net to process a single image. The FPS is the actual throughput achieved by the app.

> **NOTE:** Running these kinds of neural networks eats up a lot of battery power. The app can put a limit on the number of times per second it runs the neural net. You can change this in `setUpCamera()` by changing the line `videoCapture.fps = 50` to a smaller number.

