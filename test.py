from luna16data import LunaData
from MaskRCNN import MaskRCNN
from torch.utils.data import DataLoader
from Explorer import Explorer
import torch
import os
import cv2
import numpy

class Tester:

    def __init__(self):
        self.explorer = Explorer(True)


    def test(self):
        file = open('resultsfile.txt','w')
        i=0
        for file_name in os.listdir('labelme-luna16-Data/test'):
            print(file_name)
            image = cv2.imread(f'labelme-luna16-Data/test/{file_name}/images/img.png', 0)
            image_label = cv2.imread(f'labelme-luna16-Data/test/{file_name}/label_viz.png')
            label = cv2.imread(f'labelme-luna16-Data/test/{file_name}/masks/label.png')
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
            ret, label = cv2.threshold(label, 20, 255, cv2.THRESH_BINARY)
            boxes, mask = self.explorer.explore(image)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if not boxes.size == 0:
                for box in boxes:
                    cls = str(box[0])
                    x1 = int(box[1])
                    y1 = int(box[2])
                    x2 = int(box[3])
                    y2 = int(box[4])
                    image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    image = cv2.putText(image, cls[:5], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            file.write(file_name+'\n')
            file.write(str(int(boxes.size//5)) + '\n')
            if not boxes.size == 0:
                for box in boxes:
                    cls = str(box[0])
                    x1 = int(box[1])
                    y1 = int(box[2])
                    w = int(box[3]-box[1])
                    h = int(box[4]-box[2])
                    file.write(f'{x1} {y1} {w} {h} {cls}\n')
                # image = cv2.resize(image,None,fx=2,fy=2)
            cv2.imshow('JK', image)
            cv2.imshow('mask', mask)
            cv2.imshow('labelviz', image_label)
            cv2.imwrite(f'biaozhu/{i}.png', label)
            cv2.imwrite(f'yuche/{i}.png', mask)
            i+=1
            #if cv2.waitKey(1) == ord('c'):
                #break
        file.close()


if __name__ == '__main__':

    tester = Tester()
    tester.test()
