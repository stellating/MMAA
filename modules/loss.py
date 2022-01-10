import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        # print((-input.gather(2, target.long().unsqueeze(2)).squeeze(2)).shape)
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

def label_smoothing(inputs, epsilon=0.1):
    K = inputs.get_shape().as_list()[-1]    # number of channels
    return ((1-epsilon) * inputs) + (epsilon / K)

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps, max=1-self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps, max=1-self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-5, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(False)
                    self.xs_pos = self.xs_pos * self.targets
                    self.xs_neg = self.xs_neg * self.anti_targets
                    self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                                self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(True)
                self.loss *= self.asymmetric_w
            else:
                self.xs_pos = self.xs_pos * self.targets
                self.xs_neg = self.xs_neg * self.anti_targets
                self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                            self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)   
                self.loss *= self.asymmetric_w         
        _loss = - self.loss.sum() / x.size(0)
        _loss = _loss / y.size(1) * 1000

        return _loss
    
class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

def compute_loss(args, output, ml_out, labels, reports_ids, reports_masks):
    criterion = LanguageModelCriterion()
    ml_criterion = AsymmetricLossOptimized(
        gamma_neg=2, gamma_pos=0,
        disable_torch_grad_focal_loss=True,
        eps=1e-9,
    )
    kv_criterion = LabelSmoothing(size=output.size(-1), padding_idx=0, smoothing=0.0)
    # print(ml_out,labels)

    if args.use_MLloss and args.use_ASLloss:
        batch_size = output.size(0)
        max_length = output.size(1)
        vocab_size = output.size(2)
        # output = output.gather(2, reports_ids[:, 1:].long().unsqueeze(2)).squeeze(2)* reports_masks[:,1:]\
        reports_masks = reports_masks[:,1:].unsqueeze(2)
        reports_masks = reports_masks.expand(batch_size,max_length,vocab_size)
        output = output*reports_masks
        reports_id = reports_ids[:, 1:]
        reports = np.zeros([batch_size,max_length,vocab_size], np.float32)
        for i in range(batch_size):
            for j in range(max_length):
                labels_index = reports_id[i,j]
                reports[i,j,labels_index]=1
        reports = torch.Tensor(reports).cuda()
        loss1 = ml_criterion(output,reports)
#         loss1 = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
        loss2 = ml_criterion(ml_out,labels)
    elif args.use_MLloss:
        loss1 = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
        loss2 = ml_criterion(ml_out,labels)
    elif args.use_ASLloss:
        batch_size = output.size(0)
        max_length = output.size(1)
        vocab_size = output.size(2)
        # output = output.gather(2, reports_ids[:, 1:].long().unsqueeze(2)).squeeze(2)* reports_masks[:,1:]\
        reports_masks = reports_masks[:,1:].unsqueeze(2)
        reports_masks = reports_masks.expand(batch_size,max_length,vocab_size)
        output = output*reports_masks
        reports_id = reports_ids[:, 1:]
        reports = np.zeros([batch_size,max_length,vocab_size], np.float32)
        for i in range(batch_size):
            for j in range(max_length):
                labels_index = reports_id[i,j]
                reports[i,j,labels_index]=1
        reports = torch.Tensor(reports).cuda()
        loss1 = ml_criterion(output,reports)
#         target = reports_ids[:, 1:]
#         loss1 = kv_criterion(output.contiguous().view(-1, output.size(-1)), 
#                               target.contiguous().view(-1))
        loss2 = 0.0
    else:
        loss1 = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
        loss2 = 0.0
    return loss1,loss2
