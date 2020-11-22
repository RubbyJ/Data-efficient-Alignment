# -*- coding: utf-8 -*
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from .base_model import BaseModel
from .LayerNormLSTMCell import *
from .encoder import *
from copy import deepcopy
from .SBN import SBN1d


class AlignNet(BaseModel):
    def __init__(self, args):
        super(AlignNet, self).__init__()
        self.teacher_forcing = args.teacher_forcing
        self.actions_nc = args.actions_nc
        self.arnn_size = args.arnn_size
        self.mrnn_size = args.mrnn_size

        if args.LXMERT:
            print('* LXMERT Model *')
            self.encoder = LXMERTEncoder(args)
        else:
            if args.random_project:
                print('* Random Projection Model *')
                self.encoder = BERTRPMEncoder(args)
            else:
                print('* Full Model *')
                self.encoder = BERTEncoder(args)

        if args.dropout > 0.0:
            self.dropout = nn.Dropout(args.dropout)  # For the last Fully-Connected layers
        if args.rnn_dropout > 0:
            self.dropout_vrnn = nn.Dropout(args.rnn_dropout)
            self.dropout_srnn = nn.Dropout(args.rnn_dropout)
            self.dropout_arnn = nn.Dropout(args.rnn_dropout)
            self.dropout_mrnn = nn.Dropout(args.rnn_dropout)
        
        self.SBN = args.SBN
        if args.LN_num is None:
            self.vrnn_layer1 = nn.LSTMCell(300, 300)
            self.vrnn_layer2 = nn.LSTMCell(300, 300)
            self.srnn_layer1 = nn.LSTMCell(300, 300)
            self.srnn_layer2 = nn.LSTMCell(300, 300)

            if args.SBN:
                print('* Sequence-wise Batch Normalization *')
                self.VideoStackBN = SBN1d(300, affine=True)
                self.TextStackBN = SBN1d(300, affine=True)
            else:
                print('* No Normalization for any Stacks *')
            self.arnn_layer1 = nn.LSTMCell(self.actions_nc, self.arnn_size)
            self.arnn_layer2 = nn.LSTMCell(self.arnn_size, self.arnn_size)
            self.mrnn_layer1 = nn.LSTMCell(300 + 300, self.mrnn_size)
            self.mrnn_layer2 = nn.LSTMCell(self.mrnn_size, self.mrnn_size)
        elif (args.LN_num == 2 or args.LN_num == 4) and not args.SBN:
            self.vrnn_layer1 = LayerNormLSTMCell(300, 300)
            self.vrnn_layer2 = LayerNormLSTMCell(300, 300)
            self.srnn_layer1 = LayerNormLSTMCell(300, 300)
            self.srnn_layer2 = LayerNormLSTMCell(300, 300)

            if args.LN_num == 2:
                print('* Layer Normalization for Video/Text 2 Stacks *')
                self.arnn_layer1 = nn.LSTMCell(self.actions_nc, self.arnn_size)
                self.arnn_layer2 = nn.LSTMCell(self.arnn_size, self.arnn_size)
                self.mrnn_layer1 = nn.LSTMCell(300 + 300, self.mrnn_size)
                self.mrnn_layer2 = nn.LSTMCell(self.mrnn_size, self.mrnn_size)
            elif args.LN_num == 4:
                print('* Layer Normalization for 4 Stacks *')
                self.arnn_layer1 = LayerNormLSTMCell(self.actions_nc, self.arnn_size)
                self.arnn_layer2 = LayerNormLSTMCell(self.arnn_size, self.arnn_size)
                self.mrnn_layer1 = LayerNormLSTMCell(300 + 300, self.mrnn_size)
                self.mrnn_layer2 = LayerNormLSTMCell(self.mrnn_size, self.mrnn_size)
            else:
                raise NotImplementedError
        else:
            raise ValueError

        self.f1 = nn.Linear(600+self.mrnn_size+self.arnn_size+10, 300)
        self.f2 = nn.Linear(300, self.actions_nc)

        self.init_weights()

    def forward(self, video, sentence, vid_len, txt_len, max_seq_len, actions=None, act_len=None):
        device = video.device
        batch_size = video.shape[0]
        max_video_len, max_sent_len, max_act_len = max_seq_len

        encoding_text, encoding_video = self.encoder(sentence, video)

        voutput = torch.zeros(batch_size, max_video_len, 300, device=device)
        soutput = torch.zeros(batch_size, max_sent_len, 300, device=device)

        v_h1_t = torch.zeros(batch_size, 300, device=device)
        v_c1_t = torch.zeros(batch_size, 300, device=device)
        v_h2_t = torch.zeros(batch_size, 300, device=device)
        v_c2_t = torch.zeros(batch_size, 300, device=device)

        s_h1_t = torch.zeros(batch_size, 300, device=device)
        s_c1_t = torch.zeros(batch_size, 300, device=device)
        s_h2_t = torch.zeros(batch_size, 300, device=device)
        s_c2_t = torch.zeros(batch_size, 300, device=device)

        for j in range(max_video_len):
            v_h1_t, v_c1_t = self.vrnn_layer1(encoding_video[:, j, :], (v_h1_t, v_c1_t))
            if hasattr(self, 'dropout_vrnn'):
                v_h2_t, v_c2_t = self.vrnn_layer2(self.dropout_vrnn(v_h1_t), (v_h2_t, v_c2_t))
            else:
                v_h2_t, v_c2_t = self.vrnn_layer2(v_h1_t, (v_h2_t, v_c2_t))
            voutput[:, j, :] = v_h2_t

        for i in range(max_sent_len):
            s_h1_t, s_c1_t = self.srnn_layer1(encoding_text[:, i, :], (s_h1_t, s_c1_t))
            if hasattr(self, 'dropout_srnn'):
                s_h2_t, s_c2_t = self.srnn_layer2(self.dropout_srnn(s_h1_t), (s_h2_t, s_c2_t))
            else:
                s_h2_t, s_c2_t = self.srnn_layer2(s_h1_t, (s_h2_t, s_c2_t))
            soutput[:, i, :] = s_h2_t

        if self.SBN:
            voutput = self.VideoStackBN(voutput, vid_len)
            soutput = self.TextStackBN(soutput, txt_len)

        if actions is not None:
            out = torch.zeros(batch_size, max_act_len, self.actions_nc, device=device)
        else:
            out = []

        a_h1_t = torch.zeros(batch_size, self.arnn_size, device=device)
        a_c1_t = torch.zeros(batch_size, self.arnn_size, device=device)
        a_h2_t = torch.zeros(batch_size, self.arnn_size, device=device)
        a_c2_t = torch.zeros(batch_size, self.arnn_size, device=device)

        m_h1_t = torch.zeros(batch_size, self.mrnn_size, device=device)
        m_c1_t = torch.zeros(batch_size, self.mrnn_size, device=device)
        m_h2_t = torch.zeros(batch_size, self.mrnn_size, device=device)
        m_c2_t = torch.zeros(batch_size, self.mrnn_size, device=device)

        a_feature = torch.zeros(batch_size, self.arnn_size, device=device)
        m_feature = torch.zeros(batch_size, self.mrnn_size, device=device)

        m_h1_previous = torch.zeros(batch_size, self.mrnn_size, device=device)
        m_c1_previous = torch.zeros(batch_size, self.mrnn_size, device=device)
        m_h2_previous = torch.zeros(batch_size, self.mrnn_size, device=device)
        m_c2_previous = torch.zeros(batch_size, self.mrnn_size, device=device)
        m_h1_not_change = torch.zeros(batch_size, self.mrnn_size, device=device)
        m_c1_not_change = torch.zeros(batch_size, self.mrnn_size, device=device)
        m_h2_not_change = torch.zeros(batch_size, self.mrnn_size, device=device)
        m_c2_not_change = torch.zeros(batch_size, self.mrnn_size, device=device)

        top_v = vid_len - 1  # Cursor for Video Stack
        top_s = txt_len - 1  # Cursor for Sent Stack
        top_s_former_matches = torch.zeros(batch_size, device=device, dtype=torch.long) - 1

        self.top_s_recording = [{} for _ in range(batch_size)]

        eos = 0  # when to end the classification
        i = 0  # the i-th Action Classification
        while eos == 0:
            """ Select Current Step """
            s_feature = self.select_time_steps(soutput, top_s)  # extract the last t-th hidden feature
            v_feature = self.select_time_steps(voutput, top_v)
            p_feature = self.encoding_p_feature(top_v, top_s, vid_len, txt_len, device)
            f_feature = torch.cat([v_feature, s_feature], dim=1)

            """ Classification """
            if hasattr(self, 'dropout'):
                x = F.relu_(self.f1(self.dropout(torch.cat((f_feature, a_feature, m_feature, p_feature), 1))))
                x = self.f2(self.dropout(x))
            else:
                x = F.relu_(self.f1(torch.cat((f_feature, a_feature, m_feature, p_feature), 1)))
                x = self.f2(x)

            if actions is not None:  # Training
                # action direct A1-> At-1(before step t), without SoftMax
                out[:, i, :] = x
            else:  # Inference
                out.append(x)

            """ Determine Action """
            x = F.softmax(x, dim=1)

            if actions is None:  # Inference
                act = x.argmax(dim=1)
            elif self.teacher_forcing == 1.0:  # Training
                act = actions[:, i]
            else:
                raise ValueError

            """ Stop or not """
            if actions is not None and i >= (act_len.max() - 1):  # Training
                break
            elif actions is None:  # Inference
                eos = self.eos_pred(eos, top_v, top_s, act, self.actions_nc)
                if eos >= 1:
                    break

            """ Action Stack """
            a_input = torch.zeros_like(x).scatter_(1, act.unsqueeze(dim=1).long(), 1.0)
            # arnn input: One-hot vector
            a_h1_t, a_c1_t = self.arnn_layer1(a_input, (a_h1_t, a_c1_t))
            if hasattr(self, 'dropout_arnn'):
                a_h2_t, a_c2_t = self.arnn_layer2(self.dropout_arnn(a_h1_t), (a_h2_t, a_c2_t))
            else:
                a_h2_t, a_c2_t = self.arnn_layer2(a_h1_t, (a_h2_t, a_c2_t))
            a_feature = a_h2_t + a_h1_t

            """ Matched Stack & the Cursor moving here """
            if self.actions_nc == 2:     # one2one
                raise RuntimeError('It has gone with the wind.')
            elif self.actions_nc == 3:  # one2many
                """ Record Clip MATCHED with Sent """
                for b in range(batch_size):
                    if act[b] == 2:                # match & retain sentence
                        self.record_top_s(b, top_s[b].item(), top_v[b].item())

                """ Pop Clip """
                compare_zeros = torch.zeros_like(top_v[act == 0])
                top_v[act == 0] = torch.max(top_v[act == 0] - 1, compare_zeros)
                del compare_zeros
                """ Pop Sent """
                compare_zeros = torch.zeros_like(top_s[act == 1])
                top_s[act == 1] = torch.max(top_s[act == 1] - 1, compare_zeros)
                del compare_zeros
                """ Match but Retain Sent """
                new_match_in_batch = torch.eq(act, 2) * ~torch.eq(top_s_former_matches, top_s)
                same_match_in_batch = torch.eq(act, 2) * torch.eq(top_s_former_matches, top_s)
                assert (same_match_in_batch * new_match_in_batch == 0).all()
                """ First, if MATCH(act==2) matches the new sent, record those states """
                m_h1_previous = self.save_assign(m_h1_previous, m_h1_t, new_match_in_batch)
                m_c1_previous = self.save_assign(m_c1_previous, m_c1_t, new_match_in_batch)
                m_h2_previous = self.save_assign(m_h2_previous, m_h2_t, new_match_in_batch)
                m_c2_previous = self.save_assign(m_c2_previous, m_c2_t, new_match_in_batch)
                top_s_former_matches[new_match_in_batch] = top_s[new_match_in_batch]

                """ if MATCH the same sent as the previous, assign back to the previous state """
                m_h1_t = self.save_assign(m_h1_t, m_h1_previous, same_match_in_batch)
                m_c1_t = self.save_assign(m_c1_t, m_c1_previous, same_match_in_batch)
                m_h2_t = self.save_assign(m_h2_t, m_h2_previous, same_match_in_batch)
                m_c2_t = self.save_assign(m_c2_t, m_c2_previous, same_match_in_batch)
                assert torch.equal(top_s_former_matches[same_match_in_batch], top_s[same_match_in_batch])

                """ Move those matched ones """
                compare_zeros = torch.zeros_like(top_v[act == 2])
                top_v[act == 2] = torch.max(top_v[act == 2] - 1, compare_zeros)
                del compare_zeros

                """ Compute matched stack input on the spot """
                m_input = torch.zeros(batch_size, 600, device=device)
                for bs in range(batch_size):
                    if top_s[bs].item() in self.top_s_recording[bs]:
                        m_input[bs] = torch.cat([
                            torch.mean(voutput[bs, self.top_s_recording[bs][top_s[bs].item()]], dim=0),
                            s_feature[bs]
                        ])
                    else:
                        assert act[bs] != 2

                m_h1_t, m_c1_t = self.mrnn_layer1(m_input, (m_h1_t, m_c1_t))
                if hasattr(self, 'dropout_mrnn'):
                    m_h2_t, m_c2_t = self.mrnn_layer2(self.dropout_mrnn(m_h1_t), (m_h2_t, m_c2_t))
                else:
                    m_h2_t, m_c2_t = self.mrnn_layer2(m_h1_t, (m_h2_t, m_c2_t))

                """ Third, those unmatched states need to go back """
                m_h1_t, m_h2_t = self.back_to_non_changing(m_h1_t, m_h2_t, m_h1_not_change, m_h2_not_change, act)
                m_c1_t, m_c2_t = self.back_to_non_changing(m_c1_t, m_c2_t, m_c1_not_change, m_c2_not_change, act)
                
                m_feature = m_h2_t + m_h1_t  # i-th Step matched stack feature

                m_h1_not_change = m_h1_t
                m_h2_not_change = m_h2_t
                m_c1_not_change = m_c1_t
                m_c2_not_change = m_c2_t

            else:
                raise ValueError
            i += 1

        if actions is None:
            out = torch.stack(out, dim=1).to(device)

        return out

    @staticmethod
    def encoding_p_feature(top_v, top_s, vid_len, txt_len, device):
        top_v = top_v.float() + 1
        top_s = top_s.float() + 1

        vid_len = vid_len.float() + 1
        txt_len = txt_len.float() + 1
        p_feature = torch.zeros(top_v.shape[0], 10, device=device)

        p_feature[:, 0] = top_v
        p_feature[:, 1] = top_s
        p_feature[:, 2] = top_v / vid_len.float()
        p_feature[:, 3] = top_s / txt_len.float()
        p_feature[:, 4] = vid_len / top_v
        p_feature[:, 5] = txt_len / top_s
        p_feature[:, 6] = top_v / top_s
        p_feature[:, 7] = top_s / top_v
        p_feature[:, 8] = (vid_len - top_v) / (txt_len - top_s)
        p_feature[:, 9] = (txt_len - top_s) / (vid_len - top_v)

        return p_feature

    @staticmethod
    def save_assign(obj, to_assign, inds):
        obj_new = obj.clone()
        obj_new[inds] = to_assign[inds]
        return obj_new

    @staticmethod
    def back_to_non_changing(m_h1_t, m_h2_t, m_h1_not_c, m_h2_not_c, act):
        m_h1_t_new = m_h1_t.clone()
        m_h2_t_new = m_h2_t.clone()
        m_h1_t_new[act != 2] = m_h1_not_c[act != 2]
        m_h2_t_new[act != 2] = m_h2_not_c[act != 2]

        return m_h1_t_new, m_h2_t_new

    def record_top_s(self, index, top_s_record, top_v_record):
        """ index: index in batch size;
            top_s_record: np.array(1,); top_v_record: np.array(1,)"""
        if top_s_record not in self.top_s_recording[index].keys():
            self.top_s_recording[index].update({top_s_record: [top_v_record]})
        else:
            self.top_s_recording[index][top_s_record].append(top_v_record)


if __name__ == "__main__":
    pass
