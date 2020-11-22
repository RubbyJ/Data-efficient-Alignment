import torch
import numpy as np


def act2align(clip_id, sent_id, act, act_nc=3):
    """
    :param clip_id: tensor[ids have been reversed]
    :param sent_id: tensor[ids have been reversed]
    :param act: actions
    :param act_nc: one2one:2 ; one2many:3
    :return:
             ACT_NC 3 : dict{'sen_id': [clip_id]}
    """
    clip2sent = {}
    sent2clip = {}
    unmatched = -1

    if act_nc == 2:  # 0:M | 1:PC
        raise ValueError('It has gone with the wind.')
    elif act_nc == 3:  # 0：PC | 1: PS | 2: MRS
        i = 0
        while len(clip_id) > 0 and len(sent_id) > 0:
            if act[i] == 2:
                clip2sent[clip_id[-1].item()] = sent_id[-1].item()

                if sent_id[-1].item() not in sent2clip.keys():                 # a New sentence
                    sent2clip[sent_id[-1].item()] = [clip_id[-1].item()]
                else:                                                               # an existing sentence
                    sent2clip[sent_id[-1].item()].append(clip_id[-1].item())

                clip_id = clip_id[:-1]
            elif act[i] == 1:
                if sent_id[-1].item() not in sent2clip.keys():                 # a New sentence without any match
                    sent2clip[sent_id[-1].item()] = [unmatched]
                sent_id = sent_id[:-1]
            elif act[i] == 0:
                clip2sent[clip_id[-1].item()] = unmatched
                clip_id = clip_id[:-1]

            i = i + 1
            if i >= len(act):
                break

        if len(clip_id) > 0:
            clip_id = list(clip_id)
            clip_id.reverse()
            clip2sent.update({j.item(): unmatched for j in clip_id})
        if len(sent_id) > 0:
            if sent_id[-1].item() in sent2clip.keys():
                sent_id = sent_id[:-1]
            sent_id = list(sent_id)
            sent_id.reverse()
            sent2clip.update({k.item(): [unmatched] for k in sent_id})

    else:
        raise ValueError

    return clip2sent, sent2clip


def calculate_sequence_total_duration(seq_id, duration):
    total = 0.0
    for si in seq_id:
        total += (duration[si.item()][1] - duration[si.item()][0])
    return total


def concat_bbox(bbox_list):
    new_box_list = []

    pre_box = bbox_list[0]
    for i, bl in enumerate(bbox_list):
        if bl[0] <= pre_box[1]:
            pre_box = (min(pre_box[0], bl[0]), max(pre_box[1], bl[1]))
        else:
            new_box_list.append(pre_box)
            pre_box = bl
    if pre_box not in new_box_list:
        new_box_list.append(pre_box)
    return new_box_list


def IoU_with_bbox_list(bbox, bbox_list, return_interval=False):
    start_gt, end_gt = bbox

    inter_sum = 0.0
    union_sum = 0.0

    for box in bbox_list:
        st, en = box

        inter = min(end_gt, en) - max(start_gt, st)
        if inter > 0:
            inter_sum += inter

            start_gt = min(start_gt, st)  # Refresh the bbox for later intersection / union.
            end_gt = max(end_gt, en)
        else:  # no intersection
            union_sum += (en - st)  # If there's no intersection, current box length needs to be added to union_sum.

    if return_interval:
        return inter_sum

    # Here we add the rest length of the union bbox.
    union_sum += (end_gt - start_gt)

    iou = inter_sum / union_sum
    assert 0 <= iou <= 1

    return iou


def acc(c2s, s2c, video_ids, duration, matched_gt, unmatched_gt, video_index, action_nc=3):
    """
    :param c2s: clip2sent
    :param s2c: sent2clip
    :param video_ids: video clip ids
    :param duration: the clip's duration
    :param matched_gt: ground truth-MATCHED video segments: {sent_id: (start, end), }
    :param unmatched_gt: ground truth-UNMATCHED video segments: [ {'video': , 'containing_clips':,'bboxes': [(), ()]} ]
    :param video_index: index in unmatched_gt
    :param action_nc: one2one: 2 | one2many: 3
    :return: clip_acc, sent_acc
    """

    if action_nc == 3:
        """ Video Brunch """
        seq_whole_duration = calculate_sequence_total_duration(video_ids, duration)

        clip_acc = 0.0
        for clip_id, cor_sen_id in c2s.items():
            if cor_sen_id == -1:  # UNMATCHED Clip
                clip_acc += IoU_with_bbox_list(duration[clip_id], unmatched_gt[video_index]['bboxes'],
                                               return_interval=True)
            else:  # MATCHED Clip
                clip_acc += IoU_with_bbox_list(duration[clip_id], [matched_gt[cor_sen_id]], return_interval=True)

        clip_acc = clip_acc / seq_whole_duration

        """ Sentence Brunch """
        sent_iou = 0.0

        for sen_id, vid_list in s2c.items():
            if vid_list != [-1]:
                multi_boxes = [duration[vl] for vl in vid_list]
                multi_boxes = concat_bbox(sorted(multi_boxes, key=lambda x: x[0]))
                ith_iou = IoU_with_bbox_list(matched_gt[sen_id], multi_boxes)
                sent_iou += ith_iou

        sent_iou = sent_iou / len(s2c)

    else:
        raise ValueError

    return clip_acc, sent_iou


if __name__ == "__main__":
    pass
