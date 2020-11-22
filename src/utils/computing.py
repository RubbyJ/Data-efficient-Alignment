import torch
from .acc import *


def compute_mask_on_batch(max_seq_len, a_len):
    # a_len.Size(batch, action_len)
    batch_size = a_len.shape[0]
    mask = torch.zeros(batch_size, max_seq_len)

    for i in range(batch_size):
        mask[i, 0:a_len[i]] = 1

    return mask.reshape(-1)


def compute_loss_on_batch(x_batch, y_batch, mask_reshape, criterion):
    """
    Compute loss with prediction and real Actions.
    :param x_batch: predication actions: tensor[batch_size, act_p_len, action_nc]
    :param y_batch: ground truth actions: tensor[batch_size, seq_len]
    :param mask_reshape: result from compute_mask_on_batch
    :param criterion: Loss Criterion on CUDA with reduction='none'
    :return: loss
    """
    if x_batch.size(1) != y_batch.size(1):
        raise Exception('We only compute loss when teacher forcing !')
    action_nc = x_batch.size(2)
    loss = criterion(x_batch.view(-1, action_nc), y_batch.view(-1))
    loss = torch.sum(loss * mask_reshape) / torch.sum(mask_reshape)
    return loss


def compute_acc_on_batch(x_batch, video_ids, text_ids,
                         vid_len, txt_len,
                         seg_duration, matched_gt, unmatched_gt, video_index,
                         action_nc=3):
    """
    :param x_batch: predication actions. tensor[batch_size, seq_len]. E.g. tensor([0,2,1,1,0,2,...])
    :param video_ids: tensor[batch_size, max_seq_len]
    :param text_ids: tensor[batch_size, max_seq_len]
    :param vid_len: tensor[batch_size, 1]
    :param txt_len: tensor[batch_size, 1]
    :param seg_duration:  dict{clip_id: duration}: A dict containing the video clip duration for sent. IoU
    :param matched_gt: ground truth-MATCHED video segments: {sent_id: (start, end), }
    :param unmatched_gt: ground truth-UNMATCHED video segments: [ {'video': , 'containing_clips':,'bboxes': [(), ()]} ]
    :param video_index: index in unmatched_gt
    :param action_nc: one2many: 3
    :return: clip_acc, sent_acc
    """
    device = video_ids.device
    batch_size = x_batch.size(0)
    x_batch = x_batch.softmax(dim=2).argmax(dim=2)  # N * max_seq_len

    c_acc = 0.0
    s_acc = 0.0

    for i in range(batch_size):
        clip_id = video_ids[i][:vid_len[i]]
        sent_id = text_ids[i][:txt_len[i]]  # So, there will not be any padding sentence id later

        clip2sent, sent2clip = act2align(clip_id, sent_id, x_batch[i], act_nc=action_nc)

        clip_acc_batch, sent_acc_batch = acc(clip2sent, sent2clip, clip_id, duration=seg_duration,
                                             matched_gt=matched_gt,
                                             unmatched_gt=unmatched_gt,
                                             video_index=video_index[i])

        c_acc += clip_acc_batch
        s_acc += sent_acc_batch

    clip_acc = c_acc / batch_size
    sent_acc = s_acc / batch_size

    return torch.tensor(clip_acc).to(device), torch.tensor(sent_acc).to(device)
