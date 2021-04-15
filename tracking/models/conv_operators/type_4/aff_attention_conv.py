import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.builder import build_from_cfg_dict
from tracking.models.losses.affinity_loss import AffinityLoss
from tracking.models.losses.triplet_loss import TripletLoss

from tracking.models.registry import EDGE_REGRESSORS, CONV_OPERATORS


class AffinityAttentionLayer(nn.Module):
    """ Similar to GAT Layer, except that attention is based on the affinity matrix instead of being implicitly estimated through learnable parameters
    Reference: https://arxiv.org/abs/2006.07327 section: 3.2 Type 4
    """
    def __init__(
        self, 
        in_dim, 
        out_dim,
        **kwargs
        ):
        
        super(AffinityAttentionLayer, self).__init__()
        self.g = None
        self.adj_att = None
        
        # equation (1)
        self.fc_3 = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_4 = nn.Linear(in_dim, out_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_3.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_4.weight, gain=gain)

    def edge_attention(self, edges):
        # assign attention weights as edge embeddings
        src, dst, _ = edges.edges()
        src = src.tolist()
        dst = dst.tolist()
        attn = torch.tensor([self.adj_att[src[i]][dst[i]] for i in range(len(src))]).cuda()
        # print(src)
        # print(dst)
        # print(attn)
        return {'e': attn.reshape(-1,1)}

    def message_func(self, edges):
        # prepare messages
        return {'z': edges.src['z'], 's': edges.dst['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        
        print(nodes.nodes())
        self_feat = self.g.ndata['z'][nodes.nodes()]
        
        
        diff = nodes.mailbox['z'] - nodes.mailbox['s']
        h = self.fc_4(self_feat) + torch.sum(nodes.mailbox['e'] * self.fc_3(diff), dim=1)
        print(h.shape)
        return {'h': h}

    def _prepare_affn_attn(self, aff_mat):
        """ Prepare attentions according to the inputs
        """
        self.N, self.M = aff_mat.shape
        self.adj_att = torch.zeros((self.N+self.M, self.N+self.M))
        self.adj_att[0:self.N, self.N:self.N+self.M] = aff_mat
        self.adj_att[self.N: self.N+self.M, 0:self.N] = aff_mat.T
        self.adj_att = self.adj_att + torch.diag(torch.ones(self.N+self.M))


    def forward(self, g, aff_mat, h):
        self.g = g
        self._prepare_affn_attn(aff_mat)
        # assign embeddigns to the graph
        self.g.ndata['z'] = h
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        # add
        return self.g.ndata.pop('h')
    

@CONV_OPERATORS.register_module
class AffinityAttentionOp(nn.Module):
    
    def __init__( 
        self, 
        feature_size=128, 
        num_gnn_layers=4, 
        activation='relu',
        loss_every_layer = True,
        edge_regression = None,
        mode = 'train',
        **kwargs) -> None:

        super().__init__()
        self.mode = mode
        self.feature_size = feature_size
        self.num_gnn_layers = num_gnn_layers
        self.loss_every_layer = loss_every_layer
        self.affinity_loss = AffinityLoss()
        self.triplet_loss = TripletLoss(margin=10)
        self.edge_regr = build_from_cfg_dict(edge_regression, EDGE_REGRESSORS)

        # ASSERTIONS !
        assert self.edge_regr != None

        self.graph_conv_layers = nn.ModuleList([
            AffinityAttentionLayer(self.feature_size, self.feature_size, norm='both', weight=True, bias=True) for _ in range(self.num_gnn_layers)
        ])



    def forward(self ,graph, embeddings, gt_aff_mat, N, M):
        """
        """
        triplet_layers_loss = torch.zeros(1).cuda()
        aff_layers_loss = torch.zeros(1).cuda()
        ret_dict = {}
        
        # initialize temp aff_attention
        pred_aff_mat = torch.ones(N,M).cuda()
        # Node Aggregations
        for layer in self.graph_conv_layers:
            embeddings = layer(graph, pred_aff_mat, embeddings)
            # embeddings = self.activation(embeddings)
            if self.loss_every_layer:
                pred_aff_mat = self._regress_aff_mat(embeddings, N, M)
                if self.mode == 'train':
                    aff_layers_loss += self.affinity_loss(pred_aff_mat, gt_aff_mat)
                    triplet_layers_loss += self.triplet_loss(embeddings, gt_aff_mat)
        
        # calculate Loss based on final layer if loss every layer is not True
        if not self.loss_every_layer:
            pred_aff_mat = self._regress_aff_mat(embeddings, N, M)
            if self.mode == 'train':
                aff_layers_loss += self.affinity_loss(pred_aff_mat, gt_aff_mat)
                triplet_layers_loss += self.triplet_loss(embeddings, gt_aff_mat)
        
        if self.mode == 'train':
            # Return Loss/Total Loss Tensor
            ret_dict['total_loss'] = triplet_layers_loss + aff_layers_loss
            ret_dict['triplet_loss'] = triplet_layers_loss.detach()
            ret_dict['aff_loss'] = aff_layers_loss.detach()
            return ret_dict, pred_aff_mat
        else:
            return pred_aff_mat


    def _regress_aff_mat(self, embeddings, N, M):
        # vectorization
        edges = (embeddings[N:N+M].reshape(M,256).unsqueeze(0) - embeddings[0:N].reshape(N,256).unsqueeze(1)).reshape(N*M, 256)
        edges = self.edge_regr(edges)
        return edges.reshape(N,M) # regressed affinity matrix



if __name__ == '__main__':
    import dgl

    g = dgl.graph(([], []), num_nodes=7)
    # g = dgl.add_reverse_edges(g) # convert to undirected
    g = dgl.add_self_loop(g) # add self loops
    # print(g.num_nodes())
    
    h = torch.rand(7,2)

    aff = torch.ones(4,3)
    layer = AffinityAttentionLayer(2, 2)
    
    output = layer(g,aff, h)
    print(output)