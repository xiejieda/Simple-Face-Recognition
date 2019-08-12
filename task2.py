import numpy as np
import re,os
import cv2

def getimgdata(path,shape):
    filenames = os.listdir(path)
    data_all = []
    labels_all = []
    for i in filenames:
        filename = os.listdir(path + '/' + i)
        imgnames = []
        for j in filename:
            if re.findall('^\d+\.jpg$',j)!=[]:
                imgnames.append(j)

        n = len(imgnames)
        for j in range(n):
            img = cv2.imread(path+'/'+i+'/'+imgnames[j])  # 读取图片
            da_new = cv2.resize(img, shape)  # 压缩图片
            da_new = da_new[:, :, 0] / 255  # 颜色通道变为0，灰度图
            data_all.append(da_new)
            labels_all.append(i)
    return data_all,labels_all

if __name__=='__main__':
    path = 'faceImagesGray'
    shape = (144, 144)
    if os.path.exists(path):
        data, labels = getimgdata(path, shape)
        print(len(data), labels)
        np.save("data.npy", data)
        np.save("labels.npy", labels)
    else:
        print('未找到该文件')



