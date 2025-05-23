# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F

from torch import nn
from typing import Dict, List

from utils.misc import NestedTensor, is_main_process
# from .position_encoding import build_position_encoding

# from pytorch_pretrained_bert.modeling import BertModel
# from transformers import BertModel
from transformers import AutoTokenizer, AutoModel

"""
This code is primarily based on the MedRPG implementation from:

Chen, Zhihao et al. "Medical Phrase Grounding with Region-Phrase Context Contrastive Alignment."
MICCAI, 2023. https://arxiv.org/abs/2307.11767

Original code: https://github.com/openmedlab/MedRPG

Please refer to the original authors for core algorithmic contributions.
"""

class BERT(nn.Module):
    def __init__(self, name: str, train_bert: bool, hidden_dim: int, max_len: int, enc_num):
        super().__init__()
        # if name == 'bert-base-uncased' :
        #     self.num_channels = 768
        # else:
        #     self.num_channels = 1024
        self.num_channels = 768
        self.enc_num = enc_num

        self.bert = AutoModel.from_pretrained(name)

        if not train_bert:
            for parameter in self.bert.parameters():
                parameter.requires_grad_(False)

    def forward(self, tensor_list: NestedTensor):

        if self.enc_num > 0:
            # # pytorch_pretrained_bert version
            # all_encoder_layers, _ = self.bert(tensor_list.tensors, token_type_ids=None, attention_mask=tensor_list.mask)
            # # use the output of the X-th transformer encoder layers
            # xs = all_encoder_layers[self.enc_num - 1]

            # transformers bert version
            bert_output = self.bert(tensor_list.tensors, token_type_ids=None, attention_mask=tensor_list.mask)
            xs = bert_output.last_hidden_state
        else:
            xs = self.bert.embeddings.word_embeddings(tensor_list.tensors)

        mask = tensor_list.mask.to(torch.bool)
        mask = ~mask
        out = NestedTensor(xs, mask)

        return out

def build_bert(args):
    # position_embedding = build_position_encoding(args)
    train_bert = args.lr_bert > 0
    bert = BERT(args.bert_model, train_bert, args.hidden_dim, args.max_query_len, args.bert_enc_num)
    # model = Joiner(bert, position_embedding)
    # model.num_channels = bert.num_channels
    return bert
