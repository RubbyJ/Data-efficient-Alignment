from .larses import set_lr


def linear_warm_up(optimizer, cur_epoch, obj_lr, warm_up_length, first_factor=0.1):
    if cur_epoch < warm_up_length:
        lr_start = first_factor * obj_lr
        delta = (obj_lr - lr_start) / warm_up_length
        cur_lr = delta * cur_epoch + lr_start

        set_lr(optimizer, cur_lr)
    elif cur_epoch == warm_up_length:
        set_lr(optimizer, obj_lr)
    else:
        pass
