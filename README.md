# object_detection
this is object_detection
#这是一个图像分析的工程应用测试，此过程注意事项是文件路径，环境系统，图像尽量减少用美图和PS修改，如果是修改大小类型及其相关的名称请使用脚本
#coding = UTF-8
#1、爬取图像的脚本
import urllib.request
import re
def getHtml(url):
   #url = urllib.parse.quote(url)
   page = urllib.request.urlopen(url)
   html = page.read()
   return html

def getImg(html):
   reg = 'src="(.+?\.jpg)" alt='
   imgre = re.compile(reg)
   html = html.decode('utf-8')  # python3
   imglist = re.findall(imgre, html)
   x = 0
   for imgurl in imglist:
       urllib.request.urlretrieve(imgurl, '%s.jpg' % x)
       x+=1
   return imglist

html = getHtml("http://www.quanjing.com/category/120005/1.html")
print (getImg(html))
#2、图像的像素分布特征（直方图均衡化）
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
src=Image.open('C:/Users/Administrator/Pictures/20180917/0000243.jpg)
r,g,b=src.split()
 
plt.figure("lena")
ar=np.array(r).flatten()
plt.hist(ar, bins=256, normed=1,facecolor='r',edgecolor='r',hold=1)
ag=np.array(g).flatten()
plt.hist(ag, bins=256, normed=1, facecolor='g',edgecolor='g',hold=1)
ab=np.array(b).flatten()
plt.hist(ab, bins=256, normed=1, facecolor='b',edgecolor='b')
plt.show()
#3、旋转模糊和坐标变化增强的代码基于keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = load_img('C:/Users/Administrator/Pictures/train/0000235''.jpg')  # 这是一个PIL图像
x = img_to_array(img)  # 把PIL图像转换成一个numpy数组，形状为(3, 150, 150)
x = x.reshape((1,) + x.shape)  # 这是一个numpy数组，形状为 (1, 3, 150, 150)

# 下面是生产图片的代码
# 生产的所有图片保存在 `preview/` 目录下
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='C:/pic/', save_prefix='smoking', save_format='jpeg'):
    i += 1
    if i > 50:
        break  # 否则生成器会退出循环
        
#5、#python将视频转换为图片，格式为MP4/avi，并将其按照帧数抽取出来
import cv2
vc = cv2.VideoCapture("you_vid's_name")
c = 1
if vc.isOpened():
	rval,frame = vc.read()
else:
	rval=False
while rval:
	rval,frame=vc.read()
	cv2.imwrite('The_path_to_save_you_pices'+str(c)+'.jpg',frame)
	c=c+1
	cv2.WaitKey(1)
	vc.release()

#图片转换为视频：
import cv2
from cv2 import VideoWriter_fourcc,imread,resize
import os
img_root="the_address_of_you_pics_folder"
#edit each frame's appearing time!
fps=5
fourcc=VideoWriter_fourcc(*"MJPG")
videoWriter=cv2.VideoWriter("the_way_you_want_to_save_your_vid.avi",fourcc,fps,(1200,1200))

im_name=os.listdir(img_root)
for im_name in range(len(im_names)):
	frame=cv2.imread(img_root+str(im_name)+'.jpg')
	print im_name
	videoWriter.write(frame)
	
	videoWriter.release()
  
 #5扩增的文件目前只能扩增八倍的数据，用着voc数据上，用lableimg其他没时间测试，图像合适参数合适情况下不合适修改角度变小就行。
 #数据集扩增
import cv2
import math
import numpy as np
import xml.etree.ElementTree as ET
import os

def rotate_image(src, angle, scale=1):
    w = src.shape[1]
    h = src.shape[0]
    # 角度变弧度
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    dst = cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
    # 仿射变换
    return dst

# 对应修改xml文件
def rotate_xml(src, xmin, ymin, xmax, ymax, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    # 获取旋转后图像的长和宽
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]                                   # rot_mat是最终的旋转矩阵
    # point1 = np.dot(rot_mat, np.array([xmin, ymin, 1]))          #这种新画出的框大一圈
    # point2 = np.dot(rot_mat, np.array([xmax, ymin, 1]))
    # point3 = np.dot(rot_mat, np.array([xmax, ymax, 1]))
    # point4 = np.dot(rot_mat, np.array([xmin, ymax, 1]))
    point1 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymin, 1]))   # 获取原始矩形的四个中点，然后将这四个点转换到旋转后的坐标系下
    point2 = np.dot(rot_mat, np.array([xmax, (ymin+ymax)/2, 1]))
    point3 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymax, 1]))
    point4 = np.dot(rot_mat, np.array([xmin, (ymin+ymax)/2, 1]))
    concat = np.vstack((point1, point2, point3, point4))            # 合并np.array
    # 改变array类型
    concat = concat.astype(np.int32)
    rx, ry, rw, rh = cv2.boundingRect(concat)                        #rx,ry,为新的外接框左上角坐标，rw为框宽度，rh为高度，新的xmax=rx+rw,新的ymax=ry+rh
    return rx, ry, rw, rh

# 使图像旋转60,90,120,150,210,240,300度
imgpath = 'C:/dataset/1/'          #源图像路径
xmlpath = 'C:/dataset/2/'         #源图像所对应的xml文件路径
rotated_imgpath = 'C:/dataset/3/'
rotated_xmlpath = 'C:/dataset/4/'
for angle in (15, 30, 45, 60, 75, 90, 105, 120):
    for i in os.listdir(imgpath):
        a, b = os.path.splitext(i)                            #分离出文件名a
        img = cv2.imread(imgpath + a + '.jpg')
        rotated_img = rotate_image(img,angle)
        cv2.imwrite(rotated_imgpath + a + '_'+ str(angle) +'d.jpg',rotated_img)
        print (str(i) + ' has been rotated for '+ str(angle)+'°')
        tree = ET.parse(xmlpath + a + '.xml')
        root = tree.getroot()
        for box in root.iter('bndbox'):
            xmin = float(box.find('xmin').text)
            ymin = float(box.find('ymin').text)
            xmax = float(box.find('xmax').text)
            ymax = float(box.find('ymax').text)
            x, y, w, h = rotate_xml(img, xmin, ymin, xmax, ymax, angle)
            # cv2.rectangle(rotated_img, (x, y), (x+w, y+h), [0, 0, 255], 2)   #可在该步骤测试新画的框位置是否正确
            # cv2.imshow('xmlbnd',rotated_img)
            # cv2.waitKey(200)
            box.find('xmin').text = str(x)
            box.find('ymin').text = str(y)
            box.find('xmax').text = str(x+w)
            box.find('ymax').text = str(y+h)
        tree.write(rotated_xmlpath + a + '_'+ str(angle) +'d.xml')
        print (str(a) + '.xml has been rotated for '+ str(angle)+'°')
	
        
