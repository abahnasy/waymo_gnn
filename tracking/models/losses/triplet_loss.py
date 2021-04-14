
import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    """
    Reference: 
    * Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. https://arxiv.org/abs/1703.07737
    * https://arxiv.org/abs/2006.07327 Section 3.3
    """
    def __init__(self, margin):
        super().__init__()
        self.margin = margin
        return

    def forward(self, embeddings, gt_aff_mat):
        # calculate triplet loss
        N, M = gt_aff_mat.shape
        assert embeddings.shape == (N+M, 256) # TODO: magic number !
        # triplet loss
        triplet_loss = torch.zeros(1).cuda()
        for idx in range(M):
            track_obj_feat = embeddings[N+idx, :]
            assert track_obj_feat.shape == torch.Size([256]) # TODO: magic number
            if gt_aff_mat[:,idx].sum() == 0: #if match not found
                # log.debug("Not matched track !")
                # minimize over the second term in the triplet loss function only !
                triplet_2 = torch.min(torch.abs(track_obj_feat - embeddings[0:N, :]).sum(dim=1))
                triplet_loss = torch.max(triplet_2 + torch.tensor(self.margin), 0)[0] # return value 
                
            else: # if match found
                matched_det_idx = torch.nonzero(gt_aff_mat[:,idx], as_tuple=False).item()
                # assert matched_det.shape == 1, "Error in gt affinity matrix, cannot have more than one match" # now handled by .item() function
                matched_det_obj_feat = embeddings[matched_det_idx, :]
                unmatched_det_obj_feat = torch.cat((embeddings[0:matched_det_idx,:],embeddings[matched_det_idx+1:N,:]), dim=0)
                assert unmatched_det_obj_feat.shape[0] == N-1
                unmatched_track_obj_feat = torch.cat((embeddings[N:N+idx,:],embeddings[N+idx+1:N+M,:]), dim=0)
                assert unmatched_track_obj_feat.shape[0] == M-1
                triplet_1 = torch.abs(track_obj_feat - matched_det_idx).sum()
                
                triplet_2 = 0
                if unmatched_det_obj_feat.shape[0] != 0:
                    triplet_2 = torch.min(torch.abs(track_obj_feat - unmatched_det_obj_feat).sum(dim=1))

                triplet_3 = 0
                if unmatched_track_obj_feat.shape[0] != 0:
                    triplet_3 = torch.min(torch.abs(matched_det_obj_feat - unmatched_track_obj_feat).sum(dim=1))
                
                # print(type(matched_det_idx), matched_det_obj_feat.shape)
                # print(unmatched_det_obj_feat.shape)
                triplet_loss += torch.max((triplet_1 - triplet_2 - triplet_3 + torch.tensor(self.margin)).clamp(min=0), 0)[0] # return value
                
        return triplet_loss