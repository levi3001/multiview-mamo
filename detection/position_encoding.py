# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

#from util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        

    def forward(self, proposals):
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        proposals = torch.cat(proposals, 0)
        y_embed = (proposals[:,1]+proposals[:,3])/2
        x_embed = (proposals[:,0]+proposals[:,2])/2
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[  :, None] / dim_t
        pos_y = y_embed[  :, None] / dim_t
        pos_x = torch.stack((pos_x[  :, 0::2].sin(), pos_x[  :, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[  :, 0::2].sin(), pos_y[  :, 1::2].cos()), dim=2).flatten(1)
        pos = torch.cat((pos_y, pos_x), dim=1)
        pos = pos.split(boxes_per_image,0)
        return pos

class PositionEmbeddingSine1(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        

    def forward(self, proposals):
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        proposals = torch.cat(proposals, 0)
        y1_embed = proposals[:,1]
        x1_embed = proposals[:,0]
        y2_embed = proposals[:,3]
        x2_embed = proposals[:,2]
        dim_t = torch.arange(self.num_pos_feats//2, dtype=torch.float32, device=proposals.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x1 = x1_embed[  :, None] / dim_t
        pos_y1 = y1_embed[  :, None] / dim_t
        pos_x2 = x2_embed[  :, None] / dim_t
        pos_y2 = y2_embed[  :, None] / dim_t
        pos_x1 = torch.stack((pos_x1[  :, 0::2].sin(), pos_x1[  :, 1::2].cos()), dim=2).flatten(1)
        pos_y1 = torch.stack((pos_y1[  :, 0::2].sin(), pos_y1[  :, 1::2].cos()), dim=2).flatten(1)
        pos_x2 = torch.stack((pos_x2[  :, 0::2].sin(), pos_x2[  :, 1::2].cos()), dim=2).flatten(1)
        pos_y2 = torch.stack((pos_y2[  :, 0::2].sin(), pos_y2[  :, 1::2].cos()), dim=2).flatten(1)
        pos = torch.cat((pos_y1, pos_x1, pos_y2, pos_x2), dim=1)
        pos = pos.split(boxes_per_image,0)
        return pos

def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding