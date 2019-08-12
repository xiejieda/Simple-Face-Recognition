# Simple Face Recohnition
基于卷积神经网络的人脸识别项目
## 具体说明
### 1.人脸数据采集  
[task1.py](https://github.com/xiejieda/Simple-Face-Recognition/blob/master/task1.py)  
执行task1.py,进行人脸数据的采集，通过opencv调用摄像头拍摄人脸数据，并生成"faceImages"和"faceImagesGray"。其中"faceImages"保存的是拍摄到原图，无实际作用，而"faceImagesGray"中保存的是原图中使用haarcascade人脸特征提取器识别人脸后截下的144像素*144像素的灰度化人脸图，去除了原图中的背景等干扰。
