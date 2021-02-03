from torch.utils.data import Dataset, DataLoader
import cv2
import torch
from models.monet_s_set import Set
import numpy as np
from math import log
import random
import math

class DataSet(Dataset):

    def __init__(self, mode='train'):
        super(DataSet, self).__init__()
        print('正在初始化数据集')
        self.set = Set()
        self.image_path = self.set.image_path
        self.label_path = f'{self.set.label_path}/{mode}.txt'
        self.dataset = []
        self.image_size = self.set.image_size
        with open(self.label_path) as file:
            for line in file.readlines():
                line = line.split()
                image_name = line[0]
                path = f'{self.image_path}/{image_name}'
                image_information = []
                boxes = line[1:]
                for i in range(len(boxes) // 6):
                    box = boxes[6 * i:6 * i + 6]
                    target = int(self.set.category.index(box[0]))
                    box = [float(j) for j in box[1:]]
                    c_x,c_y,w,h,angle = box
                    image_information.append((target, c_x, c_y, w, h, angle))
                self.dataset.append([path, image_information])
                print(image_information)
        print('数据集初始化完成')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        if self.set.is_mosaic:
            image, boxes = self.mosaic(item)
        else:
            image_path, boxes = self.dataset[item]
            image = cv2.imread(image_path)
            h,w,c = image.shape
            # print(image.shape)
            max_len = max(h,w)
            fx = self.set.image_size/max_len
            image = cv2.resize(image,None,fx=fx,fy=fx)
            h,w,c = image.shape
            ground = np.zeros((640,640,3),dtype=np.uint8)
            # print(h,w,c,fx)
            ground[320-h//2:320-h//2+h,320-w//2:320-w//2+w] = image
            image = ground
            boxes_new = []
            for box in boxes:
                target, c_x, c_y, _w, _h, angle = box
                c_x = c_x * fx - w // 2 + 320
                c_y = c_y * fx - h // 2 + 320
                _w = _w * fx
                _h = _h * fx
                boxes_new.append((target, c_x, c_y, _w, _h, angle))

                '''检查标签转换正确否'''
                # x1 = c_x + (_w/2)*math.cos(angle) - (_h/2)*math.sin(angle)
                # y1 = c_y + (_w / 2) * math.sin(angle) + (_h / 2) * math.cos(angle)
                #
                # x2 = c_x + (-_w / 2) * math.cos(angle) - (_h / 2) * math.sin(angle)
                # y2 = c_y + (-_w / 2) * math.sin(angle) + (_h / 2) * math.cos(angle)
                #
                # x3 = c_x + (-_w / 2) * math.cos(angle) - (-_h / 2) * math.sin(angle)
                # y3 = c_y + (-_w / 2) * math.sin(angle) + (-_h / 2) * math.cos(angle)
                #
                # x4 = c_x + (_w / 2) * math.cos(angle) - (-_h / 2) * math.sin(angle)
                # y4 = c_y + (_w / 2) * math.sin(angle) + (-_h / 2) * math.cos(angle)
                #
                # pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                # pts = pts.reshape((-1, 1, 2))
                # # 如果第三个参数为False，将获得连接所有点的折线，而不是闭合形状。
                # img = cv2.polylines(image, [pts], True, (0, 0, 255),2)
                # cv2.imshow('a',image)
            cv2.waitKey()
            boxes = boxes_new

        '''图片、标签，制作网络能用的数据形式'''
        image_tensor = torch.from_numpy(image).float() / 255
        image_tensor = image_tensor.permute(2, 0, 1)

        targets_13, targets_26, targets_52 = self.make_target(boxes)
        targets_13 = torch.from_numpy(targets_13)
        targets_26 = torch.from_numpy(targets_26)
        targets_52 = torch.from_numpy(targets_52)
        return image_tensor, targets_13, targets_26, targets_52

    def mosaic(self, item):
        back_ground = np.zeros((640, 640, 3))
        reticle_w, reticle_h = random.randint(10, 630), random.randint(10, 630)
        boxes = []
        '''剪裁第一张图'''
        item = random.randint(0, len(self.dataset) - 1)
        image_path, boxes = self.dataset[item]
        image = cv2.imread(image_path)
        # back_ground[:reticle_h,:reticle_w] =
        return back_ground, boxes

    def make_target(self, boxes):
        targets_13 = np.zeros((self.image_size // 32, self.image_size // 32, self.set.anchor_num, 9), dtype=np.float32)
        targets_26 = np.zeros((self.image_size // 16, self.image_size // 16, self.set.anchor_num, 9), dtype=np.float32)
        targets_52 = np.zeros((self.image_size // 8, self.image_size // 8, self.set.anchor_num, 9), dtype=np.float32)
        '''循环每一个标签框，放入'''
        for box in boxes:
            target = box[0]
            c_x, c_y, w, h, angle = box[1:]
            c_x, c_y, w, h, alpha1, alpha2, gamma = self.reset_box(c_x, c_y, w, h, angle)

            i = 0
            iou = 0
            trunk = []

            '''循环测试该目标框和哪一个目标更加的匹配，iou的值最大'''
            for size in self.set.boxes_base:
                # print(size)
                stride = self.set.image_size // size
                index_h = c_y // stride
                index_w = c_x // stride
                offset_x = (c_x % stride) / stride
                offset_y = (c_y % stride) / stride
                for box2 in self.set.boxes_base[size]:
                    ratio_w = w / box2[0]
                    ratio_h = h / box2[1]
                    if i == 0:
                        trunk = [int(index_h), int(index_w), 1., offset_x, offset_y, log(ratio_w), log(ratio_h), alpha1, alpha2, gamma, target, 0, 0]
                        iou = self.calculate_iou((w, h), box2)
                    else:
                        next_iou = self.calculate_iou((w, h), box2)
                        if next_iou > iou:
                            iou = next_iou
                            trunk = [int(index_h), int(index_w), 1., offset_x, offset_y, log(ratio_w), log(ratio_h),
                                     alpha1, alpha2, gamma,
                                     target, i // len(self.set.boxes_base[size]),
                                     i % len(self.set.boxes_base[size])]
                    i += 1

            '''写入标签中'''
            # print(w,h)
            # print(trunk[12])
            if trunk[11] == 0:
                targets_52[trunk[0], trunk[1], trunk[-1]] = torch.tensor(trunk[2:11])
            elif trunk[11] == 1:
                targets_26[trunk[0], trunk[1], trunk[-1]] = torch.tensor(trunk[2:11])
            elif trunk[11] == 2:
                targets_13[trunk[0], trunk[1], trunk[-1]] = torch.tensor(trunk[2:11])
        return targets_13, targets_26, targets_52

    @staticmethod
    def calculate_iou(box1, box2):
        min_w = min(box1[0], box2[0])
        min_h = min(box1[1], box2[1])
        intersection = min_w * min_h
        area1 = box1[0] * box2[0]
        area2 = box1[1] * box2[1]
        return intersection / (area1 + area2 - intersection)

    @staticmethod
    def reset_box(c_x, c_y, w, h, angle):
        """重新写，改为c_x,c_y,u,v,s,p"""
        '''c_x,c_y,u,v,s,p 太复杂，还是换四个点序，吧'''
        # print(c_x,c_y,h,w,angle)

        '''计算四个点的向量'''
        x1 = (w / 2) * math.cos(angle) - (h / 2) * math.sin(angle)
        y1 = (w / 2) * math.sin(angle) + (h / 2) * math.cos(angle)
        x2 = (-w / 2) * math.cos(angle) - (h / 2) * math.sin(angle)
        y2 = (-w / 2) * math.sin(angle) + (h / 2) * math.cos(angle)
        x3 = (-w / 2) * math.cos(angle) - (-h / 2) * math.sin(angle)
        y3 = (-w / 2) * math.sin(angle) + (-h / 2) * math.cos(angle)
        x4 = (w / 2) * math.cos(angle) - (-h / 2) * math.sin(angle)
        y4 = (w / 2) * math.sin(angle) + (-h / 2) * math.cos(angle)

        if angle == 0:
            alpha1, alpha2, gamma = 0,0,1
            _w, _h = w, h
        else:
            x_points = {x1: y1, x2: y2, x3: y3, x4: y4}
            y_points = {y1: x1, y2: x2, y3: x3, y4: x4}
            _x1 = min(x1, x2, x3, x4)
            _y1 = min(y1, y2, y3, y4)
            _x2 = max(x1, x2, x3, x4)
            _y2 = max(y1, y2, y3, y4)
            _w, _h = _x2 - _x1, _y2 - _y1
            # c_x, c_y = (_x1 + _x2) / 2, (_y1 + _y2) / 2
            s1 = y_points[_y1] - _x1
            s2 = x_points[_x2] - _y1
            alpha1 = s1 / _w
            alpha2 = s2 / _h
            gamma = (w*h) / (_w * _h)

        # print(alpha1, alpha2, gamma)
        return c_x, c_y, _w, _h, alpha1, alpha2, gamma


if __name__ == '__main__':
    voc = DataSet()
    voc[1]
    # for i in range(len(voc)):
    #     voc[i]
