import torch
import math


def update_adam_lars(optimizer, lars_coef):
    """

    Example:
        parameters = [{'params': p} for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)

        for i in range(iter_num):
            ...

            optimizer.zero_grad()
            loss.backward()

            ori_lr = update_adam_lars(optimizer, lars_coef)
            # The base learning rate for each 'param_group' in optimizer is the same by default,
            # or you may customize your own version for resetting the different base learning rates.
            optimizer.step()
            set_lr(optimizer, ori_lr)

    """
    ori_lr = optimizer.param_groups[0]['lr']
    for group in optimizer.param_groups:
        assert len(group['params']) == 1
        for p in group['params']:
            if p.grad is None:
                continue

            p_norm = torch.norm(p.data)
            grad = p.grad.data

            state = optimizer.state[p]

            # State 0
            if len(state) == 0:
                step = 0
                # Exponential moving average of gradient values
                exp_avg = torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                exp_avg_sq = torch.zeros_like(p.data)
            else:
                step = state['step']
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

            beta1, beta2 = group['betas']
            bias_correction1 = 1 - beta1 ** (step + 1)
            bias_correction2 = 1 - beta2 ** (step + 1)

            if group['weight_decay'] != 0:
                grad = grad + group['weight_decay'] * p.data

            numer = (exp_avg * beta1 + (1 - beta1) * grad) / bias_correction1
            denom = (exp_avg_sq * beta2 + (1 - beta2) * grad * grad).sqrt() / math.sqrt(bias_correction2) + group['eps']

            global_lr = group['lr']

            if p_norm * torch.norm(numer / denom) == 0:
                local_lr = lars_coef
            else:
                local_lr = lars_coef * p_norm / torch.norm(numer / denom)
            group['lr'] = global_lr * local_lr

            if len(state) != 0:
                assert (exp_avg, exp_avg_sq) == (state['exp_avg'], state['exp_avg_sq'])
                assert step == state['step']

    return ori_lr


def set_lr(optimizer, ori_lr):
    for group in optimizer.param_groups:
        assert len(group['params']) == 1
        group['lr'] = ori_lr

