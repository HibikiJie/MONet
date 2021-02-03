from xml.etree import cElementTree as ET
import cv2
import os
import numpy

def catch_information():
    """解析xml文件为txt"""
    path = 'outputs'
    images_path = 'image'
    targets = []
    for file_name in os.listdir(path):
        file = f'{path}/{file_name}'
        tree = ET.parse(file)
        name = tree.findtext('filename')
        image_path = f'{images_path}/{name}'
        image = cv2.imread(image_path)
        massage = name+'.jpg'
        print(file)
        for obj in tree.iter('object'):
            print(obj.findtext('robndbox'))
            x1 = obj.findtext('robndbox/cx')
            y1 = obj.findtext('robndbox/cy')
            x2 = obj.findtext('robndbox/w')
            y2 = obj.findtext('robndbox/h')
            angle = obj.findtext('robndbox/angle')
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            angle= float(angle)
            # pts = numpy.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], numpy.int32)
            # 顶点个数：4，矩阵变成4*1*2维
            # pts = pts.reshape((-1, 1, 2))

            # image = cv2.polylines(image, [pts],True,(0,0,255),2,)
            name = obj.findtext('name')
            # if name not in targets:
            #     targets.append(name)
            massage = f'{massage}  {name}  {x1}  {y1}  {x2}  {y2}  {angle}'
        print(massage)
        with open('train.txt', 'a') as f:
            f.write(massage + '\n')
        # cv2.imshow('JK', image)
        # if cv2.waitKey() == ord('c'):
        #     continue


catch_information()
