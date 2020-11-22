import torch.utils.data as data
import os
import numpy as np


class NeuMatch_YMS_LXMERT_Dataset(data.Dataset):
    def __init__(self, root_dir, ds_name, max_seq_len, phase):
        super(NeuMatch_YMS_LXMERT_Dataset, self).__init__()
        if phase not in ['training', 'val', 'test']:
            raise ValueError('Phase are 3 types: training, val and test.')
        self.phase = phase
        path = os.path.join(root_dir, ds_name+'_origin_pixel_dimnorm')

        self.vid_embedding = np.load(os.path.join(path, 'lxmert_features/video', 'one_norm_full_lxmert_video.npy'),
                                     allow_pickle=True).item()
        self.bert_embedding = np.load(os.path.join(path, 'lxmert_features/sentence', 'one_norm_full_lxmert_sent.npy'),
                                      allow_pickle=True).item()

        self.videos = np.load(os.path.join(path, phase,
                                           'yms_{}_cut'.format(phase)+'_100'*(phase == 'training')+'_video.npy'),
                              allow_pickle=True)
        self.texts = np.load(os.path.join(path, phase,
                                          'yms_{}_cut'.format(phase)+'_100'*(phase == 'training')+'_sent.npy'),
                             allow_pickle=True)
        if phase == 'training':
            self.actions_set = np.load(os.path.join(path, phase,
                                                    'yms_{}_cut'.format(phase)+'_100'*(phase == 'training')+'_action.npy'),
                                       allow_pickle=True)

        if phase == 'training':
            self.videos_index_in_unmatched = np.load(os.path.join(
                path, phase, 'yms_{}_cut_100_video_index_seq_in_unmatched.npy'.format(phase)),
                allow_pickle=True)
        else:
            self.videos_index_in_unmatched = np.load(os.path.join(
                path, phase, 'yms_{}_cut_video_index_seq_in_unmatched.npy'.format(phase)),
                allow_pickle=True)

        self.max_video_len, self.max_sent_len, self.max_act_len = max_seq_len
        self.dims = (768, 768)

    def __getitem__(self, index):
        # video
        video_ids = list(self.videos[index])
        video_ids.reverse()
        vid_len = len(video_ids)

        video = np.zeros((self.max_video_len, self.dims[0]), dtype=np.float32)
        for i in range(vid_len):
            video[i] = self.vid_embedding[video_ids[i]]

        # sent
        text_ids = list(self.texts[index])
        text_ids.reverse()
        txt_len = len(text_ids)

        sentence = np.zeros((self.max_sent_len, self.dims[1]), dtype=np.float32)
        for i in range(txt_len):
            sentence[i] = self.bert_embedding[text_ids[i]]

        if self.phase == 'training':
            actions = np.array(self.actions_set[index]).astype(np.int64)

            act_len = len(actions)
            actions = np.pad(actions, (0, self.max_act_len - act_len), 'constant', constant_values=1)
        else:
            actions = None
            act_len = None

        video_ids = np.pad(video_ids, (0, self.max_video_len - vid_len),
                           mode='constant', constant_values=-1)
        text_ids = np.pad(text_ids, (0, self.max_sent_len - txt_len),
                          mode='constant', constant_values=-1)

        video_index = self.videos_index_in_unmatched[index]

        if self.phase == 'training':
            return {'video_ids': video_ids, 'text_ids': text_ids, 'video': video,
                    'sentence': sentence,
                    'actions': actions,  'vid_len': vid_len, 'txt_len': txt_len,
                    'act_len': act_len, 'video_index': video_index}
        else:
            return {'video_ids': video_ids, 'text_ids': text_ids, 'video': video,
                    'sentence': sentence, 'vid_len': vid_len, 'txt_len': txt_len,
                    'video_index': video_index}

    def __len__(self):
        return len(self.videos)


if __name__ == '__main__':
    pass
