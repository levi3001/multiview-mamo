# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.
Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from detection.multihead_attention import MultiheadAttention 

class CrossviewTransformer(nn.Module):

    def __init__(self, d_model=1024, nhead=8, num_encoder_layers=1,
                 num_decoder_layers=1, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, use_self_attn = False, compute_attn=False ):
        super().__init__()
        self.use_self_attn =use_self_attn

        decoder_layer0 = nn.ModuleList([TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, use_self_attn, compute_attn) for i in range(num_decoder_layers)])
        decoder_layer1 = nn.ModuleList([TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, use_self_attn, compute_attn) for i in range(num_decoder_layers)])
        decoder_layer = nn.ModuleList([decoder_layer0, decoder_layer1])
        decoder_norm1 = nn.LayerNorm(d_model)
        decoder_norm2 = nn.LayerNorm(d_model)
        self.decoder = TwoviewTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm1, decoder_norm2,
                                        return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, roi_CC, roi_MLO, CC_key_padding_mask, MLO_key_padding_mask, MLO_pos, CC_pos ):
        roi_CC, roi_MLO = self.decoder(roi_CC, roi_MLO, CC_key_padding_mask= CC_key_padding_mask , MLO_key_padding_mask = MLO_key_padding_mask,
                        MLO_pos= MLO_pos, CC_pos= CC_pos)
        return roi_CC, roi_MLO



class TwoviewTransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm1=None, norm2 = None, return_intermediate=False):
        super().__init__()
        #self.layers = _get_clones(decoder_layer, num_layers)
        self.layers = decoder_layer
        self.num_layers = num_layers
        self.norm1 = norm1
        self.norm2 = norm2
        self.return_intermediate = return_intermediate

    def forward(self, roi_CC, roi_MLO,
                CC_mask: Optional[Tensor] = None,
                MLO_mask: Optional[Tensor] = None,
                CC_key_padding_mask: Optional[Tensor] = None,
                MLO_key_padding_mask: Optional[Tensor] = None,
                MLO_pos: Optional[Tensor] = None,
                CC_pos: Optional[Tensor] = None):

        if self.norm1 is not None:
            roi_CC = self.norm1(roi_CC)
            roi_MLO = self.norm2(roi_MLO)
            
        for i in range(self.num_layers):
            layer_CC_MLO= self.layers[0][i]
            layer_MLO_CC = self.layers[1][i]
            roi_CC1 = layer_CC_MLO(roi_CC, roi_MLO, tgt_mask=CC_mask,
                        memory_mask=MLO_mask,
                        tgt_key_padding_mask=CC_key_padding_mask,
                        memory_key_padding_mask=MLO_key_padding_mask,
                        pos=MLO_pos, query_pos=CC_pos)
            #h= layer_CC_MLO.linear1.weight.register_hook(lambda grad: print(grad is not None, i) )
            roi_MLO1 = layer_MLO_CC(roi_MLO, roi_CC, tgt_mask=MLO_mask,
                        memory_mask=CC_mask,
                        tgt_key_padding_mask=MLO_key_padding_mask,
                        memory_key_padding_mask=CC_key_padding_mask,
                        pos=CC_pos, query_pos=MLO_pos)
            roi_CC = roi_CC1
            roi_MLO = roi_MLO1
        return roi_CC, roi_MLO




class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1,
                 activation="relu", normalize_before=False, use_self_attn =False, compute_attn=False):
        super().__init__()
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first= True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.use_self_attn = use_self_attn
        if self.use_self_attn:
            self.self_attn =  nn.MultiheadAttention(d_model, nhead, dropout=dropout,  batch_first= True)
            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.compute_attn= compute_attn
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor+pos

    def forward(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        
        if self.use_self_attn:
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
        
        
        tgt2, attn2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask, average_attn_weights=False)
        if self.compute_attn:
            self.attn= attn2
            self.tgt = tgt
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt




def _get_clones(module, N):
    return nn.ModuleList([nn.ModuleList([copy.deepcopy(module) for i in range(N)]), nn.ModuleList([copy.deepcopy(module) for i in range(N)])])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")