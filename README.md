# Advance Perception in Autonomous Vehicles

The proposed system gives the Autonomous vehicle the ability to sense the road in which it also keeps track of the lane in which it is driving ,all by using computer vision as a source of perception. The three-dimensional occupancy of other cars and bicycles on the road can be obtained using OpenCV only through camera frame systems with speed estimation. <br/>

## 1. Install all dependencies
```
python install -r requirement.txt
```

## 2. Download
1. [Yolov4-tiny-weights](https://drive.google.com/file/d/1yJVFZShj9YD-Bysq5P5ZNBphjRJzpr9p/view?usp=share_link)
and paste in <br/>
SpeedOfCar  <br/>
  |- model_data <br/>

2. [YoloV3-weights](https://drive.google.com/drive/folders/1xDQSCmEtx9RVDGNn0USBruAhSzTznhrm?usp=share_link) for 3D bounding box
and paste in 3D-boundingBox folder<br/>

## 3. Run the code
To run code go to <br/>
SpeedOfCar <br/>
  |- main.py <br/>
and run the command
```
python main.py
```

### Lane segmentation is being separated from main code it is folder Segmentation folder.

To run code go to <br/>
Segmentation <br/>
  |- testing.py <br/>
and run the command
```
python testing.py
```
