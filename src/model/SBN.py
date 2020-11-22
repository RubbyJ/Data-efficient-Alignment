import torch
import torch.distributed as dist
import math
import random
from torch.nn.utils.rnn import pad_sequence
from torch.nn.modules.batchnorm import _NormBase


class SBN1d(_NormBase):
    # nn.BatchNorm records the unbiased variance but computes with biased variance
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, var_unbiased=True):
        super(SBN1d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.var_unbiased = var_unbiased

    def forward(self, input, real_len):
        """
        :param input: Tensor-Size (N, padding-Channel, dim)
        :param real_len: Tensor-Size (N), the real length along the padded channel
        """
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        new_mean, new_var = self.batch_nonpadding_computation1d(input, real_len)

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

            self.running_mean = (1 - exponential_average_factor) * self.running_mean \
                                + exponential_average_factor * new_mean
            self.running_var = (1 - exponential_average_factor) * self.running_var \
                               + exponential_average_factor * new_var
        else:
            if self.running_mean is not None:  # Inference when tracking
                new_mean = self.running_mean
                new_var = self.running_var

        res = (input - new_mean) / torch.sqrt(new_var + self.eps)

        if self.affine:
            res = res * self.weight + self.bias

        return res

    def batch_nonpadding_computation1d(self, input, real_len):
        """
        This function computes the mean of those non-padding features.
        :param input: Tensor-Size (N, padding-Channel, dim)
        :param real_len: Tensor-Size (N), the real length along the padded channel
        """
        if len(real_len.shape) != 1 or real_len.shape[0] != self.num_features != input.size(-1):
            raise ValueError('expected 3-D input and 1-D real_len with num_features {}, '
                             'but {} input and {} real_len'.format(self.num_features,
                                                                   input.size(), real_len.shape))

        N, C, dimension = input.size()
        non_padding_len = torch.sum(real_len, dtype=torch.float)
        # padding_len = N * C - non_padding_len
        non_padding_list = [input[n, :real_len[n], :] for n in range(N)]
        non_padding_tensor = torch.cat(non_padding_list, dim=0)
        assert non_padding_tensor.size(0) == non_padding_len

        non_padding_var, non_padding_mean = torch.var_mean(non_padding_tensor, dim=0, unbiased=self.var_unbiased)

        return non_padding_mean, non_padding_var

    def reset_running_stats_momentum_at_last(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.zero_()
            self.num_batches_tracked.zero_()
            self.momentum = None


if __name__ == '__main__':
    import numpy as np
    # np.random.seed(10)
    # random.seed(14)
    a = torch.ones(4, 2, 3, requires_grad=True)
    a[0, 1, :] = 0
    real_len = torch.tensor([1, 2, 2, 2], dtype=torch.int)
    pbn = SBN1d(3, affine=True, var_unbiased=False, track_running_stats=True)

    print('pbn 1:', pbn(a, real_len)[0])
    print(pbn.running_var)
    print('pbn 2:', pbn(a, real_len)[0])
    print(pbn.running_var)
    print('pbn 3:', pbn(a, real_len)[0])
    print(pbn.running_var)
    pbn.eval()
    print('pbn eval:', pbn(a, real_len)[0])
    print(pbn.running_var)
    print('_____')
    # nn.BatchNorm Records the the unbiased variance but computes with biased variance
    bn = torch.nn.BatchNorm1d(3, affine=True, track_running_stats=True)
    print('bn 1:', bn(a.reshape(-1, 3))[:2])
    print(bn.running_var)
    print('bn 2:', bn(a.reshape(-1, 3))[:2])
    print(bn.running_var)
    print('bn 3:', bn(a.reshape(-1, 3))[:2])
    print(bn.running_var)
    bn.eval()
    print('bn eval:', bn(a.reshape(-1, 3))[:2])
    print(bn.running_var)
