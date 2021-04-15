import torch
import torch.nn as nn

from tools.builder import build_from_cfg_dict
from tracking.models.losses.affinity_loss import AffinityLoss
from tracking.models.losses.triplet_loss import TripletLoss

from tracking.models.registry import EDGE_REGRESSORS, CONV_OPERATORS

class MessagePassingLayer(nn.Module):
    """ Reference: https://arxiv.org/abs/2006.07327 section: 3.2 Type 1
    """
    def __init__(
        self, 
        in_dim, 
        out_dim,
        **kwargs
        ):
        
        super(MessagePassingLayer, self).__init__()
        self.g = None
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.activation = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)


    def message_func(self, edges):
        # prepare messages
        return {'z': edges.src['z']}

    def reduce_func(self, nodes):

        h = torch.sum(self.activation(self.fc(nodes.mailbox['z'])), dim=1)
        return {'h': h}


    def forward(self, g, h):
        self.g = g
        
        # assign embeddigns to the graph
        
        self.g.ndata['z'] = h
        self.g.update_all(self.message_func, self.reduce_func)
        # add
        return self.g.ndata.pop('h')

@CONV_OPERATORS.register_module
class MessagePassingOp(nn.Module):
    
    def __init__( 
        self, 
        feature_size=128, 
        num_gnn_layers=4, 
        activation='relu',
        loss_every_layer = True,
        edge_regression = None,
        mode = 'train',
        **kwargs
        ) -> None:

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
            MessagePassingLayer(self.feature_size, self.feature_size, norm='both', weight=True, bias=True) for _ in range(self.num_gnn_layers)
        ])
        self.activation = nn.ReLU() if activation == 'relu' else nn.Tanh()



    def forward(self ,graph, embeddings, gt_aff_mat, N, M):
        """
        """
        triplet_layers_loss = torch.zeros(1).cuda()
        aff_layers_loss = torch.zeros(1).cuda()
        ret_dict = {}
        
        # initialize temp aff_attention
        # Node Aggregations
        for layer in self.graph_conv_layers:
            embeddings = layer(graph, embeddings)
            embeddings = self.activation(embeddings)
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

    g = dgl.graph(([0,1,2,3], [4,5,6,7]), num_nodes=8)
    # g = dgl.add_reverse_edges(g) # convert to undirected
    g = dgl.add_self_loop(g) # add self loops
    # print(g.num_nodes())
    
    h = torch.randn(8,2)

    aff = torch.ones(4,3)
    layer = MessagePassingLayer(2, 2)
    
    output = layer(g, h)
    print(output)