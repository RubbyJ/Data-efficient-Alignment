import torch
import time
import numpy as np
from src.utils.computing import *
from src.solver.larses import *
from src.solver.warm_up import linear_warm_up
from src.utils.tools import AverageMeter


def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    device = torch.cuda.current_device()
    train_loss = 0.0
    train_clip_acc = AverageMeter()
    train_text_acc = AverageMeter()

    len_train = len(train_loader)
    model.train()

    torch.cuda.synchronize()
    start = time.time()

    assert optimizer.param_groups[0]['lr'] == optimizer.param_groups[1]['lr']
    if epoch >= args.warm_up_length:
        print('Epoch {} lr: {}'.format(epoch, optimizer.param_groups[0]['lr']))
    for i, batch in enumerate(train_loader):
        if args.warm_up:
            linear_warm_up(optimizer, epoch + float(i) / len_train, args.lr, args.warm_up_length)
            if epoch < args.warm_up_length:
                assert optimizer.param_groups[0]['lr'] == optimizer.param_groups[1]['lr'] == optimizer.param_groups[2]['lr']
                print('Epoch {} Iter {} lr: {}'.format(epoch, i, optimizer.param_groups[0]['lr']))

        video = batch['video'].to(device)
        sentence = batch['sentence'].to(device)

        video_ids = batch['video_ids'].to(device)
        text_ids = batch['text_ids'].to(device)
        vid_len = batch['vid_len'].to(device)
        txt_len = batch['txt_len'].to(device)

        actions = batch['actions'].to(device)
        act_len = batch['act_len'].to(device)

        video_index = batch['video_index']

        pred_a = model(video, sentence,
                       vid_len, txt_len,
                       args.training_max_seq_len,
                       actions=actions, act_len=act_len)

        mask_reshape = compute_mask_on_batch(actions.size(-1), act_len)

        loss = compute_loss_on_batch(pred_a, actions, mask_reshape.to(device), criterion)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)

        """ LARS """
        if args.adamlars:
            ori_lr = update_adam_lars(optimizer, args.lars_coef)

            optimizer.step()

            set_lr(optimizer, ori_lr)
        else:
            optimizer.step()

        clip_acc, text_acc = compute_acc_on_batch(pred_a, video_ids, text_ids, vid_len, txt_len,
                                                  video_index=video_index,
                                                  seg_duration=args.duration,
                                                  matched_gt=args.matched_gt,
                                                  unmatched_gt=args.unmatched_gt,
                                                  action_nc=args.actions_nc)
        train_clip_acc.update(clip_acc.item(), video.size(0))
        train_text_acc.update(text_acc.item(), video.size(0))

    train_loss /= len_train

    torch.cuda.synchronize()
    end = time.time()

    print('Train {:3d} epoch, Training Loss: {:.6f}, Clip Acc: {:.6f}, Text Acc: {:.6f}, '
          'Time:{:.2g}s'.format(epoch + 1, train_loss, train_clip_acc.avg, train_text_acc.avg, end - start))
    writer.add_scalars('Train', {'Loss': train_loss, 'Clip Acc': train_clip_acc.avg,
                                 'Text Acc': train_text_acc.avg}, (epoch + 1))

    return train_loss
