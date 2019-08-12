# Simple Face Recohnition
基于卷积神经网络的人脸识别项目
## 具体说明
### 1.人脸数据采集  
[task1.py](https://github.com/xiejieda/Simple-Face-Recognition/blob/master/task1.py)  
执行task1.py,进行人脸数据的采集，通过opencv调用摄像头拍摄人脸数据，并生成"faceImages"和"faceImagesGray"。其中"faceImages"保存的是拍摄到原图，无实际作用。而"faceImagesGray"中保存的是原图中使用haarcascade人脸特征提取器识别人脸后截下的144像素*144像素的灰度化人脸图，去除了原图中的背景等干扰。 
### 2.图片数据整理
[task2.py](https://github.com/xiejieda/Simple-Face-Recognition/blob/master/task2.py)  
执行task2.py，进行数据整理，将"faceImagesGray"中图片数据整理成numpy中的ndarray格式，并保存为"data.npy"和"labels.npy"。  
### 3.模型构建与测试  
[task3.py](https://github.com/xiejieda/Simple-Face-Recognition/blob/master/task3.py)  
执行task3.py，将"data.npy"和"labels.npy"按8:2分为训练集与测试集，使用训练集对模型进行训练，并对训练完成之后的模型按测试准确率高低进行保存。  
### 4.模型应用  
[task4.py](https://github.com/xiejieda/Simple-Face-Recognition/blob/master/task4.py)  
执行task4.py，调用task3.py中保存好的模型，实现实时对对抓拍到的人脸照片进行识别，或者传入人脸照片进行识别。
