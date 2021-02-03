from torch import nn
import torch


class FocalLoss(nn.Module):

    def __init__(self, gamma=2, alpha=0.25, r=1e-19):
        """
        :param gamma: gamma>0减少易分类样本的损失。使得更关注于困难的、错分的样本。越大越关注于困难样本的学习
        :param alpha:调节正负样本比例
        :param r:数值稳定系数。
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce_loss = nn.BCELoss()
        self.r = r

    def forward(self, p, target):
        target = target.float()
        p_min = p.min()
        p_max = p.max()
        if p_min < 0 or p_max > 1:
            raise ValueError('The range of predicted values should be [0, 1]')
        p = p.reshape(-1, 1)
        target = target.reshape(-1, 1)
        loss = -self.alpha * (1 - p) ** self.gamma * (target * torch.log(p + self.r)) - \
               (1 - self.alpha) * p ** self.gamma * ((1 - target) * torch.log(1 - p + self.r))
        return loss.mean()


class FocalLossManyClassification(nn.Module):

    def __init__(self, num_class, alpha=None, gamma=2, smooth=None, epsilon=1e-19):
        """
        FocalLoss,适用于多分类。输入带有softmax，无需再softmax。
        :param num_class: 类别数。
        :param alpha: 各类别权重系数，输入列表，长度需要与类别数相同。
        :param gamma: 困难样本学习力度
        :param smooth: 标签平滑系数
        :param epsilon: 数值稳定系数
        """
        super(FocalLossManyClassification, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, list):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('Smooth value should be in [0,1]')
        self.epsilon = epsilon

    def forward(self, input_, target):
        '''softmax激活'''
        logit = torch.softmax(input_, dim=1)

        if logit.dim() > 2:
            raise ValueError('The input dimension should be 2')
        target = target.reshape(-1, 1)

        alpha = self.alpha
        if alpha.device != input_.device:
            alpha = alpha.to(input_.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.epsilon
        log_pt = pt.log()

        alpha = alpha[idx]
        loss = -1 * alpha * ((1 - pt) ** self.gamma) * log_pt

        return loss.mean()


if __name__ == '__main__':
    f = FocalLossManyClassification(10, alpha=[1, 2, 15, 4, 8, 6, 7, 7, 9, 4], smooth=0.1)
    predict = torch.randn(64, 10, requires_grad=True)
    targets = torch.randint(0, 9, (64,))
    loss = f(torch.sigmoid(predict), targets)
    print(loss)
    loss.backward()
    # print(targets)
