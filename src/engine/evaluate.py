import torch
import time
import numpy as np
from src.utils.computing import *
from src.utils.tools import AverageMeter
import os


def evaluate(eval_loader, model, args, writer, epoch, phase):
    if phase not in ['Validate', 'Test']:
        raise ValueError('Only 2 mode for evaluate function: Validate & Test.')

    device = torch.cuda.current_device()
    model.eval()
    eval_clip_acc = AverageMeter()
    eval_text_acc = AverageMeter()

    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            video = batch['video'].to(device)
            sentence = batch['sentence'].to(device)
            video_ids = batch['video_ids'].to(device)
            text_ids = batch['text_ids'].to(device)
            vid_len = batch['vid_len'].to(device)
            txt_len = batch['txt_len'].to(device)

            video_index = batch['video_index']

            pred_a = model(
                video,
                sentence,
                vid_len,
                txt_len,
                args.max_seq_len
                )

            clip_acc, text_acc = compute_acc_on_batch(
                pred_a, video_ids, text_ids, vid_len, txt_len,
                seg_duration=args.duration,
                matched_gt=args.matched_gt,
                unmatched_gt=args.unmatched_gt,
                video_index=video_index,
                action_nc=args.actions_nc)

            eval_clip_acc.update(clip_acc.item(), video.size(0))
            eval_text_acc.update(text_acc.item(), video.size(0))

    torch.cuda.synchronize()
    end = time.time()

    print('{} Clip Acc: {:.6f}, Text Acc: {:.6f}, Time:{:.2g}s'.format(phase,
          eval_clip_acc.avg, eval_text_acc.avg, end - start))
    if writer is not None and epoch is not None:
        writer.add_scalars('{}'.format(phase), {'Clip Acc': eval_clip_acc.avg,
                                                'Text Acc': eval_text_acc.avg},
                           (epoch + 1))

    return eval_clip_acc.avg, eval_text_acc.avg
