import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.center_utils import _transpose_and_gather_feat

class RegLoss(nn.Module):
  '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  def __init__(self):
    super(RegLoss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.float().unsqueeze(2) 

    loss = F.l1_loss(pred*mask, target*mask, reduction='none')
    loss = loss / (mask.sum() + 1e-4)
    loss = loss.transpose(2 ,0).sum(dim=2).sum(dim=1)
    return loss

class FastFocalLoss(nn.Module):
  '''
  Reimplemented focal loss, exactly the same as the CornerNet version.
  Faster and costs much less memory.
  '''
  def __init__(self):
    super(FastFocalLoss, self).__init__()

  def forward(self, out, target, ind, mask, cat):
    '''
    Arguments:
      out, target: B x C x H x W
      ind, mask: B x M
      cat (category id for peaks): B x M
    '''
    mask = mask.float()
    gt = torch.pow(1 - target, 4) # -ve predictions, not targets, but consider nearyby and empty cells
    neg_loss = torch.log(1 - out) * torch.pow(out, 2) * gt #(p^2)log(1-p)
    neg_loss = neg_loss.sum()

    pos_pred_pix = _transpose_and_gather_feat(out, ind) # B x M x C
    pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2)) # B x M
    num_pos = mask.sum()
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * \
               mask.unsqueeze(2)
    pos_loss = pos_loss.sum()
    if num_pos == 0:
      return - neg_loss
    return - (pos_loss + neg_loss) / num_pos


class MyCrossEntropyLoss(nn.Module):
  def __init__(self):
    super(MyCrossEntropyLoss, self).__init__()
    self.crit = nn.CrossEntropyLoss(weight=torch.tensor([0.4932, 1.0531, 43.3120])) # {'VEHICLE': 0.49324813200998713, 'PEDESTRIAN': 1.053147559635907, 'CYCLIST': 43.31202097620008}

  def forward(self, out, target, ind, mask, cat):
    
    gt = torch.pow(1 - target, 4) # -ve predictions, not targets, but consider nearyby and empty cells
    neg_loss = torch.log(1 - out) * torch.pow(out, 2) * gt #(p^2)log(1-p)
    neg_loss = - neg_loss.sum()

    pos_pred_pix = _transpose_and_gather_feat(out, ind) # B x M x C
    pos_gt_pix = _transpose_and_gather_feat(target, ind)
    pos_gt = pos_gt_pix.gather(2, cat.unsqueeze(2)) # B x M
    pos_gt = pos_gt.squeeze(2)
    pos_gt = pos_gt.view(-1,1).long().squeeze(1)
    
    
    
    
    loss = self.crit(pos_pred_pix.view(-1, 3), pos_gt)
    # pos_pred_pix = _transpose_and_gather_feat(out, ind) # B x M x C
    # print(pos_pred_pix.shape); print(mask.sum()); exit()
    # loss = target*torch.log(out) + (1-target)*torch.log(1-out)
    # loss = loss.sum()
    
    
    return loss + neg_loss


