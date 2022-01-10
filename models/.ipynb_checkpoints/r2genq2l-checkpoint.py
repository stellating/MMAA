import torch
import torch.nn as nn
import numpy as np
import math

from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder
from lib.models.query2label import build_q2l

class GroupWiseLinear(nn.Module):
    # could be changed to:
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.label_extractor = build_q2l(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        # self.fc = nn.Linear(args.num_class*2,args.num_class)
        if args.dataset_name == 'thyroid':
            self.forward = self.forward_thyroid
        elif args.dataset_name == 'iu_xray':
            self.forward = self.forward_iuxray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_thyroid(self, images, targets=None, mode='train'):
        image_feature, hidden_state, fc_feature = self.label_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(image_feature, hidden_state, fc_feature, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feature, image_feature, hidden_state, mode='sample')
        else:
            raise ValueError
        return fc_feature, output

    def forward_iuxray(self, images, targets=None, mode='train'):
        image_feature, hidden_state, fc_feature = self.label_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(image_feature, hidden_state, fc_feature, targets, mode='forward')
        elif mode == 'sample':
            output, out_prob = self.encoder_decoder(fc_feature, image_feature, hidden_state, mode='sample')
        else:
            raise ValueError
        return fc_feature, output

    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        # att_feats, fc_feats = self.visual_extractor(images)
        image_feature, hidden_state, fc_feature = self.label_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(image_feature, hidden_state, fc_feature, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feature, image_feature, hidden_state, mode='sample')
        else:
            raise ValueError
        return fc_feature,output


