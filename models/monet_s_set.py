import numpy


class Set:

    def __init__(self):
        '''图片和标签的位置'''
        self.image_path = '/home/cq/pubilic/hibiki/MONet/data/image'
        self.label_path = '/home/cq/pubilic/hibiki/MONet/data'
        self.image_size = 640  # 图片尺寸
        size = 's'
        depth_multiple = None
        width_multiple = None
        '''设置锚定框的尺寸大小'''
        self.boxes_base = {
            self.image_size//8: ((10, 13), (16, 30), (33, 23)),
            self.image_size//16: ((30, 61), (64, 45), (59, 119)),
            self.image_size//32: ((116, 90), (156, 198), (373, 326))
        }
        self.anchor_num = len(self.boxes_base.keys())
        self.boxes_base2 = {
            self.image_size//8: numpy.array([[10, 13], [16, 30], [33, 23]], dtype=numpy.float32),
            self.image_size//16: numpy.array([[30, 61], [62, 45], [59, 119]], dtype=numpy.float32),
            self.image_size//32: numpy.array([[116, 90], [156, 198], [373, 326]], dtype=numpy.float32)
        }
        self.threshold = 0.8  # 目标置信度
        self.iou_threshold = 0.1  # iou阈值
        self.gamma_threshold = 0.8
        """分类类别"""
        self.category = ['person', 'moniter', 'bird', 'shap', 'plane', 'train']
        self.num_category = len(self.category)
        self.is_mosaic = False
        if size == 's':
            self.depth_multiple = 0.33  # model depth multiple
            self.width_multiple = 0.50  # layer channel multiple
        elif size == 'm':
            self.depth_multiple = 0.67
            self.width_multiple = 0.75
        elif size == 'l':
            self.depth_multiple = 1.0
            self.width_multiple = 1.0
        elif size == 'x':
            self.depth_multiple = 1.33
            self.width_multiple = 1.25
        else:
            if depth_multiple is None or width_multiple is None:
                raise ValueError('Depth_multiple or depth_multiple is None,please check.')
            self.depth_multiple = depth_multiple
            self.width_multiple = width_multiple


if __name__ == '__main__':
    s = Set()
    print(s.boxes_base)