import torch
import math


def random_projection_matrix(orign_cn, object_cn):
    frame_projection_matrix = torch.randint(0, 6, (orign_cn, object_cn))
    frame_projection_matrix[frame_projection_matrix == 0] = -1
    frame_projection_matrix[frame_projection_matrix > 1] = 0
    frame_projection_matrix = frame_projection_matrix.type(torch.float32)
    frame_projection_matrix = frame_projection_matrix * math.sqrt(3)

    return frame_projection_matrix

