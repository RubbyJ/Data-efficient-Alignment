import torch
import torch.nn as nn


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, reduction='mean'):
        super(LabelSmoothingLoss, self).__init__()
        assert 0.0 <= smoothing <= 1.0
        smoothing_value = smoothing / classes

        self.confidence = 1.0 - smoothing + smoothing_value

        smooth_one_hot = torch.full((classes,), smoothing_value)
        self.register_buffer('smooth_one_hot', smooth_one_hot)
        self.reduction = reduction

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)

        assert pred.size()[:-1] == target.size()
        assert len(target.size()) == 1

        smooth_tgt = self.smooth_one_hot.repeat(target.size(0), 1)
        smooth_tgt.scatter_(1, target.unsqueeze(1), self.confidence)

        if self.reduction == 'mean':
            loss = torch.mean(torch.sum(- smooth_tgt * pred, dim=-1))
        elif self.reduction == 'sum':
            loss = torch.sum(- smooth_tgt * pred)
        elif self.reduction == 'none':
            loss = torch.sum(- smooth_tgt * pred, dim=-1)
        else:
            raise RuntimeError('LabelSmoothingLoss reduction only supports \'none\',' 
                               '\'sum\', \'mean\' now.')

        return loss


if __name__ == '__main__':
    pass


