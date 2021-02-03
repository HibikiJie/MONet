from utils.datasets import DataSet
from models.monet import MONet
from torch.utils.data import DataLoader
from utils.loss import FocalLoss,FocalLossManyClassification
from torch import nn
import torch
import os


class Trainer:

    def __init__(self, Net=MONet, load_parameters=False, is_distributed=False):
        self.is_distributed = is_distributed
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        print('准备使用设备%s训练网络' % self.device)

        '''实例化模型，并加载参数'''
        self.net = Net().train()
        if os.path.exists('weights/net.pth') and load_parameters:
            self.net.load_state_dict(torch.load('weights/net.pth'))
            print('成功加载模型参数')
        elif not load_parameters:
            print('未能加载模型参数')
        else:
            raise RuntimeError('Model parameters are not loaded')
        self.net = self.net.to(self.device)
        print('模型初始化完成')

        '''实例化数据集，并实例化数据加载器'''
        self.data_set = DataSet()
        self.data_loader = DataLoader(self.data_set, len(self.data_set), True, num_workers=2)

        '''实例化损失函数'''
        self.f_loss = FocalLoss(2, 0.8)
        self.f_loss_mc = FocalLossManyClassification(6)
        # self.f_loss_mc = nn.CrossEntropyLoss()
        self.mse_loss = torch.nn.MSELoss().to(self.device)
        print('损失函数初始化完成')

        '''实例化优化器'''
        self.optimizer = torch.optim.Adam(self.net.parameters())
        if os.path.exists('optimizer.pth') and load_parameters:
            self.optimizer.load_state_dict(torch.load('optimizer.pth'))
            print('成功加载训练器参数')
        elif load_parameters:
            raise Warning('未能正确加载优化器参数')
        else:
            print('优化器初始化完成')
        self.sigmoid = torch.nn.Sigmoid()

        '''开启分布式训练'''
        if is_distributed:
            self.net = nn.parallel.DataParallel(self.net)

    def train(self):
        i = 0
        epoch = 0  # 训练轮次
        accumulation_steps = 1  # 梯度累积步数
        self.net.train()
        loss_min = 1e20
        while True:
            loss_sum = 0
            loss1_sum = 0
            loss2_sum = 0
            loss3_sum = 0
            for images, targets_16, targets_32, targets_64 in self.data_loader:

                if self.is_distributed:
                    images = images.cuda()
                    targets_16 = targets_16.cuda()
                    targets_32 = targets_32.cuda()
                    targets_64 = targets_64.cuda()
                else:
                    images = images.to(self.device)
                    targets_16 = targets_16.to(self.device)
                    targets_32 = targets_32.to(self.device)
                    targets_64 = targets_64.to(self.device)

                p64, p32, p16 = self.net(images)
                '''计算损失'''
                loss1 = self.compute_loss(p64, targets_64)
                loss2 = self.compute_loss(p32, targets_32)
                loss3 = self.compute_loss(p16, targets_16)
                loss = loss1 + loss2 + loss3

                '''反向传播，梯度更新'''
                # loss.backward()
                # if (i + 1) % accumulation_steps == 0:
                #     self.optimizer.step()
                #     self.optimizer.zero_grad()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                '''统计损失信息'''
                i += 1
                loss1_sum += loss1.item()
                loss2_sum += loss2.item()
                loss3_sum += loss3.item()
                loss_sum += loss.item()
            epoch += 1
            '''写日志文件'''
            logs = f'''{epoch},loss_sum: {loss_sum / len(self.data_loader)},loss_64:{loss1_sum / len(self.data_loader)},loss_32:{loss2_sum / len(self.data_loader)},loss_16:{loss3_sum / len(self.data_loader)} '''
            print(logs)
            with open('logs.txt', 'a') as file:
                file.write(logs + '\n')

            '''保存模型参数'''
            if (loss_sum / len(self.data_loader)) < loss_min:
                loss_min = loss_sum / len(self.data_loader)
                if self.is_distributed:
                    torch.save(self.net.module.state_dict(), 'weights/net_best.pth')
                else:
                    torch.save(self.net.state_dict(), 'weights/net_best.pth')
            else:
                if self.is_distributed:
                    torch.save(self.net.module.state_dict(), 'weights/net.pth')
                else:
                    torch.save(self.net.state_dict(), 'weights/net.pth')
                torch.save(self.optimizer.state_dict(), 'weights/optimizer.pth')

    def compute_loss(self, predict, target):

        """标签形状为（N,H,W,3,6）"""
        mask_positive = target[:, :, :, :, 0] > 0.5
        mask_negative = target[:, :, :, :, 0] < 0.5

        target_positive = target[mask_positive]
        target_negative = target[mask_negative]
        number, _ = target_positive.shape
        predict_positive = predict[mask_positive]
        predict_negative = predict[mask_negative]

        '''置信度损失'''
        if number > 0:
            loss_c_p = self.f_loss(self.sigmoid(predict_positive[:, 0]), target_positive[:, 0])
            loss_gamma_p = self.f_loss(self.sigmoid(predict_positive[:, 7]), target_positive[:, 7])
        else:
            loss_c_p = 0
            loss_gamma_p = 0
        loss_c_n = self.f_loss(self.sigmoid(predict_negative[:, 0]), target_negative[:, 0])
        loss_gamma_n = self.f_loss(self.sigmoid(predict_negative[:, 7]), target_negative[:, 7])
        loss_c = loss_c_n + loss_c_p + loss_gamma_p + loss_gamma_n

        '''边框回归'''
        if number > 0:
            loss_box1 = self.mse_loss(1.1*self.sigmoid(predict_positive[:, 1:3]-0.05), target_positive[:, 1:3])
            loss_box2 = self.mse_loss(predict_positive[:, 3:5], target_positive[:, 3:5])
            loss_alpha = self.mse_loss(self.sigmoid(predict_positive[:, 5:7]), target_positive[:, 5:7])
            '''分类损失'''
            # print(predict_positive[:, 5:].shape)
            loss_class = self.f_loss_mc(predict_positive[:, 8:], target_positive[:, 8].long())

        else:
            loss_box1 = 0
            loss_box2 = 0
            loss_alpha = 0
            loss_class = 0
        return loss_c + (loss_box1 + loss_box2 + loss_alpha) + loss_class


if __name__ == '__main__':
    trainer = Trainer(load_parameters=False, is_distributed=False)
    trainer.train()
