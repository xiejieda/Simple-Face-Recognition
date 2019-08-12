import tensorflow as tf
import cv2


ID = ("baozhongxin","liucheng","mayouyou", "xiejieda")

w = 144
h = 144
c = 1
# -----------------构建网络----------------------
# 占位符
x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')

def CNNlayer():
    # 第一个卷积层
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # 第二个卷积层
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # 第三个卷积层
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # 第四个卷积层
    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    re1 = tf.reshape(pool4, [-1, 9 * 9 * 128])

    # 全连接层
    dense1 = tf.layers.dense(inputs=re1,
                             units=1024,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    dense2 = tf.layers.dense(inputs=dense1,
                             units=512,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    logits = tf.layers.dense(inputs=dense2,
                             units=4,
                             activation=None,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    return logits

logits = CNNlayer()
predict = tf.argmax(logits, 1)

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, 'model/faces-6')


user = input('图片（G）还是摄像头（V）:')
if user == 'G':
    path = input('图片路径名是：')
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (w, h))
    img = img.reshape([w, h, c])
    res = sess.run(predict, feed_dict={x: [img]})
    print(ID[res[0]])


else:
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    # 告诉OpenCV使用人脸识别分类器
    classfier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    # 识别出人脸后要画的边框的颜色，RGB格式, color是一个不可增删的数组
    color = (0, 255, 0)
    # 视频封装格式
    while cap.isOpened():
        ok, frame = cap.read()  # 读取一帧数据
        if not ok:
            break
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将当前桢图像转换成灰度图像
        # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(128, 128),
                                               maxSize=(128, 128))
        if len(faceRects) > 0:  # 大于0则检测到人脸
            for faceRect in faceRects:  # 单独框出每一张人脸
                x1, y1, w1, h1 = faceRect
                img = grey[y1 - 10: y1 + h1 + 10, x1 - 10: x1 + w1 + 10]
                try:
                    img = cv2.resize(img, (w, h))
                    img = img.reshape([w, h, c])
                    res = sess.run ( predict , feed_dict={x: [img]} )
                    print(res)
                    # 画出矩形框
                    cv2.rectangle ( frame , (x1 - 10 , y1 - 10) , (x1 + w1 + 10 , y1 + h1 + 10) , color , 2 )
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText ( frame , ID[res[0]] , (x1 + 30 , y1 + 30) , font , 1 , (255 , 0 , 255) , 2 )
                except Exception:
                    pass
            # 显示图像v
        cv2.imshow('Capture', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()
sess.close()