import torch
import torch.nn as nn

class AffinityLoss(nn.Module):
    """
    Reference:
    * https://arxiv.org/abs/2006.07327 Section 3.3
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()
        

    def forward(self, pred_aff_mat, gt_aff_mat):
        """
        """
        N, M = gt_aff_mat.shape
        aff_loss =  self.bce(pred_aff_mat.view(-1,1).unsqueeze(0), gt_aff_mat.view(-1,1).unsqueeze(0).float())
        
        # calculations on NxM
        not_matched_indices = torch.where(gt_aff_mat.sum(dim=1))[0].tolist()
        match_indices = [i for i in range(N) if i not in not_matched_indices]
        if len(match_indices) != 0:
            aff_loss +=  self.ce(pred_aff_mat[match_indices, :], gt_aff_mat[match_indices,:].argmax(dim=1))
        
        # calculations on MxN
        not_matched_indices = torch.where(gt_aff_mat.sum(dim=0))[0].tolist()
        match_indices = [i for i in range(M) if i not in not_matched_indices]
        if len(match_indices) != 0:
            aff_loss +=  self.ce(pred_aff_mat[:, match_indices].T, gt_aff_mat[:, match_indices].argmax(dim=0).T)

        return aff_loss