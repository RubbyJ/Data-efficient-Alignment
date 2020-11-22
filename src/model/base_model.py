import torch
import torch.nn as nn
from torch.nn import init


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    @staticmethod
    def select_time_steps(tensor, length_indices):

        batch = []
        for i, index in enumerate(length_indices):
            batch.append(tensor[i][index])
        g = torch.stack(batch)

        return g

    def init_weights(self, init_param=0.1):
        def set_forget_gate(l):
            '''
            set forget gate bias to 1
            '''
            n = l.bias_ih.size(0)
            start, end = n // 4, n // 2
            l.bias_ih.data[start:end].fill_(1.)
            n = l.bias_hh.size(0)
            start, end = n // 4, n // 2
            l.bias_hh.data[start:end].fill_(1.)

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and classname.find('Linear') != -1:
                init.kaiming_normal_(m.weight.data)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.normal_(m.bias, std=init_param)
            elif classname.find('LSTMCell') != -1:
                init.xavier_normal_(m.weight_ih)
                init.orthogonal_(m.weight_hh)
                if hasattr(m, 'bias_ih'):
                    init.normal_(m.bias_hh, std=init_param)
                    init.normal_(m.bias_ih, std=init_param)
                    set_forget_gate(m)

        self.apply(init_func)

    def print(self):
        num_params = 0
        for name, param in self.named_parameters():
            num_params += param.numel()
            # print(name, param.numel())
        print(self)
        print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))

    @staticmethod
    def eos_pred(eos, top_v, top_s, err_act, actions_nc):  # Inference eos
        # This function is for prediction
        if actions_nc == 2:
            if (top_v*top_s == 0).all():
                eos = eos + 1  # top_s : 0 top_v : 1
                assert eos == 1
        elif actions_nc == 3:
            if (top_v*top_s == 0).all():
                bsz = err_act.size(0)
                eos_batch = [0] * bsz
                for i in range(bsz):
                    if top_v[i] == 0:
                        if err_act[i] != 1:  # 1 :pop sentence
                            eos_batch[i] = 1
                    if top_s[i] == 0:
                        if err_act[i] == 1:
                            eos_batch[i] = 1

                if sum(eos_batch) == bsz:
                    eos = eos + 1
                    assert eos == 1
        else:
            raise ValueError
        return eos
