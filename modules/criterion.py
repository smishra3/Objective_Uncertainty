import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs), targets)

#https://towardsdatascience.com/multi-task-learning-with-pytorch-and-fastai-6d10dc7ce855
class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num, model):
        super(MultiTaskLossWrapper, self).__init__()
        self.model = model
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))
        self.weight = torch.tensor([1/0.9105,1/0.0895])

    def forward(self, inputs, targets, targets_map):

        preds = self.model(inputs)
        loss0 = nn.CrossEntropyLoss(self.weight.cuda())(preds[0], targets)
        #loss1 = nn.CrossEntropyLoss(self.weight.cuda())(preds[1], targets)
      
        (bsz,H,W) = targets.size()
        logp = F.log_softmax(preds[1])
        logp = logp.gather(1, targets.view(bsz,1,H,W))

        #weights = torch.cuda.FloatTensor(bsz,1,H,W).random_(1,3)
        weights = targets_map
     
        weighted_logp = (logp * weights).view(bsz, -1)
        weighted_loss = weighted_logp.sum(1)/weights.view(bsz, -1).sum(1)
        loss1 = -1 * weighted_loss.mean()
        
        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0*loss0 + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1*loss1 + self.log_vars[1]


        return loss0+loss1, self.log_vars.data.tolist()
        

class MultiTaskLossWrapper_wo(nn.Module):
    def __init__(self, task_num, model):
        super(MultiTaskLossWrapper, self).__init__()
        self.model = model
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))
        self.weight = torch.tensor([1/0.9105,1/0.0895])

    def forward(self, input, targets):

        preds = self.model(input)
        loss0 = nn.CrossEntropyLoss(self.weight.cuda())(preds[0], targets)
        loss1 = nn.CrossEntropyLoss(self.weight.cuda())(preds[1], targets)

        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0*loss0 + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1*loss1 + self.log_vars[1]

        return loss0+loss1, self.log_vars.data.tolist()


class MultiTaskLossWrapperi_vanilla(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))
        self.weight = torch.tensor([1/0.9105,1/0.0895])

    def forward(self, preds, targets):
        loss0 = nn.CrossEntropyLoss(self.weight.cuda())(preds[0], targets)
        loss1 = nn.CrossEntropyLoss(self.weight.cuda())(preds[1], targets)

        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0*loss0 + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1*loss1 + self.log_vars[1]

        return loss0+loss1, self.log_vars.data.tolist()


def jaccard_loss(true, logits, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)
