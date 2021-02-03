from models.monet import MONet
from models.monet_s_set import Set
import torch
import numpy
import cv2
from time import time
import os


class Explorer(object):

    def __init__(self, is_cuda=False):
        self.set = Set()
        self.device = torch.device('cuda:1' if torch.cuda.is_available() and is_cuda else 'cpu')
        self.net = MONet().to(self.device)
        self.net.load_state_dict(torch.load('weights/net_best.pth',map_location='cpu'))
        self.net.eval()

    def explore(self, input_):
        start = time()
        input_ = torch.from_numpy(input_).float() / 255
        input_ = input_.permute(2, 0, 1).unsqueeze(0)
        input_ = input_.to(self.device)  # 数据传入处理设备

        '''模型预测'''
        predict1, predict2, predict3 = self.net(input_)
        print('cost_time1:', time() - start)
        boxes1, category1 = self.select(predict1.detach().cpu().numpy(), 8, self.set.image_size//8)
        boxes2, category2 = self.select(predict2.detach().cpu().numpy(), 16, self.set.image_size//16)
        boxes3, category3 = self.select(predict3.detach().cpu().numpy(), 32, self.set.image_size//32)
        boxes = self.stack_box(boxes1,boxes2,boxes3)
        category = numpy.hstack([category1,category2,category3])
        print(boxes,category)

        boxes, category = self.nms_with_category(boxes, category)
        print('cost_time2:',time()-start)
        return boxes, category

    def select(self, predict, len_side, fm_size):
        """
        通过阈值筛选，并且完成反算。传输参数为array（N,C,H,W）.

        参数：
            predict: 预测值
        返回:

        """
        '''n,h,w,3,15'''
        n, h, w, c, _ = predict.shape

        '''挑选置信度大于阈值的数据,获取位置索引'''
        predict[:, :, :, :, 0] = self.sigmoid(predict[:, :, :, :, 0])
        index = numpy.where(predict[:, :, :, :, 0] > self.set.threshold)
        print(index)
        # exit()
        '''获取宽高，框类索引'''
        index_h = index[1]
        index_w = index[2]
        box_base = index[3]

        '''通过索引，索引出数据'''
        boxes_with_category = predict[index]

        '''反算回原图'''
        c_x = (index_w + 1.1*self.sigmoid(boxes_with_category[:, 1])-0.05) * len_side
        c_y = (index_h + 1.1*self.sigmoid(boxes_with_category[:, 2])-0.05) * len_side

        w = numpy.exp(boxes_with_category[:, 3]) * self.set.boxes_base2[fm_size][box_base][:, 0]
        h = numpy.exp(boxes_with_category[:, 4]) * self.set.boxes_base2[fm_size][box_base][:, 1]

        x1 = c_x - w / 2
        y1 = c_y - h / 2

        x2 = x1 + w
        y2 = y1 + h

        '''计算所属类别'''
        category = boxes_with_category[:, 8:]
        category = numpy.argmax(category, axis=1)

        '''旋转信息'''
        alpha1 = 1.1*self.sigmoid(boxes_with_category[:, 5])-0.05
        alpha2 = 1.1*self.sigmoid(boxes_with_category[:, 6])-0.05
        gamma = self.sigmoid(boxes_with_category[:,7])

        '''返回边框和类别信息'''
        return numpy.stack((boxes_with_category[:, 0], x1, y1, x2, y2, alpha1,alpha2, gamma), axis=1), category

    def nms_with_category(self, boxes, categorys):
        if boxes.size == 0:
            return numpy.array([]), numpy.array([])
        """根据类别的不同，进行非极大值抑制"""
        picked_boxes = []
        picked_category = []
        for category_index in range(self.set.num_category):
            '''索引该类别的数据'''
            index = categorys == category_index
            box1 = boxes[index]
            '''排序'''
            order = numpy.argsort(box1[:, 0])[::-1]
            box1 = box1[order]

            while box1.shape[0] > 1:
                max_box = box1[0]
                picked_boxes.append(max_box)
                picked_category.append(numpy.array([category_index]))
                left_box = box1[1:]
                iou = self.calculate_iou(max_box, left_box)
                index = iou < self.set.iou_threshold
                box1 = left_box[index]
            if box1.shape[0] > 0:
                picked_boxes.append(box1[0])
                picked_category.append(numpy.array([category_index]))

        return numpy.vstack(picked_boxes), numpy.hstack(picked_category)

    @staticmethod
    def calculate_iou(box1, box2):
        area1 = (box1[3] - box1[1]) * (box1[4] - box1[2])
        area2 = (box2[:, 3] - box2[:, 1]) * (box2[:, 4] - box2[:, 2])

        x1 = numpy.maximum(box1[1], box2[:, 1])
        y1 = numpy.maximum(box1[2], box2[:, 2])
        x2 = numpy.minimum(box1[3], box2[:, 3])
        y2 = numpy.minimum(box1[4], box2[:, 4])

        intersection_area = numpy.maximum(0, x2 - x1) * numpy.maximum(
            0, y2 - y1)
        return intersection_area / numpy.minimum(area1, area2)

    @staticmethod
    def sigmoid(x):
        x = torch.from_numpy(x)
        x = torch.sigmoid(x)
        return x.numpy()

    @staticmethod
    def stack_box(*boxes):
        boxes_new = []
        for box in boxes:
            if box.size == 0:
                continue
            else:
                boxes_new.append(box)
        if len(boxes_new) == 0:
            return numpy.array([[]])
        else:
            boxes_new = numpy.vstack(boxes)
        return boxes_new
if __name__ == '__main__':
    explorer = Explorer(False)
    set1 = Set()
    for file_name in os.listdir('data/image'):
        s = time()
        image = cv2.imread(f'data/image/{file_name}')
        h, w, c = image.shape
        # print(image.shape)
        max_len = max(h, w)
        fx = 640/ max_len
        image = cv2.resize(image, None, fx=fx, fy=fx)
        h, w, c = image.shape
        ground = numpy.zeros((640, 640, 3), dtype=numpy.uint8)
        # print(h,w,c,fx)
        ground[320 - h // 2:320 - h // 2 + h, 320 - w // 2:320 - w // 2 + w] = image
        image = ground
        boxes = explorer.explore(image)
        print(boxes)
        for box, index in zip(boxes[0], boxes[1]):
            name = set1.category[index]
            x1 = int(box[1])
            y1 = int(box[2])
            x2 = int(box[3])
            y2 = int(box[4])
            # print(box)
            # exit()
            #

            image = cv2.putText(image, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            alpha1, alpha2,gamma = box[5:]
            if gamma >0.8:
                image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:
                w,h = x2-x1,y2-y1
                _x1,_y1 = x1+w*alpha1,y1
                _x2,_y2 = x2,y1+h*alpha2
                _x3, _y3 = x2-w*alpha1,y2
                _x4,_y4 = x1,y2-h*alpha2

                pts = numpy.array([[_x1, _y1], [_x2, _y2], [_x3, _y3], [_x4, _y4]], numpy.int32)
                pts = pts.reshape((-1, 1, 2))
                image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                image = cv2.polylines(image, [pts], True, (0, 0, 255),2)


            # image = cv2.resize(image,None,fx=2,fy=2)
        cv2.imshow('JK', image)
        if cv2.waitKey(0) == ord('c'):
            continue
