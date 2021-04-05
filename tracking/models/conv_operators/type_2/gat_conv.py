import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv

from tracking.models.registry import CONV_OPERATORS

@CONV_OPERATORS.register_module
class GATConvOp(nn.Module):
    """ https://arxiv.org/pdf/1710.10903.pdf
    """
    
    def __init__(
        self, 
        feature_size=128, 
        num_gnn_layers=4, # not used here, 
        activation='relu',
        gat_conv_num_head = 3,
        gat_conv_feat_drop =  0.0,
        gat_conv_attn_drop =  0.0,
        **kwargs
        ):

        super(GATConvOp, self).__init__()
        self.feature_size = feature_size
        self.num_gnn_layers = num_gnn_layers
        
        
        
        self.attentions = GATConv(
            self.feature_size, 
            self.feature_size, 
            num_heads = gat_conv_num_head, 
            feat_drop= gat_conv_feat_drop, 
            attn_drop= gat_conv_attn_drop, 
            negative_slope=0.2, 
            residual=False, 
            activation=None, 
            allow_zero_in_degree=False
        )
        self.out_att = GATConv(
            self.feature_size * gat_conv_num_head, 
            self.feature_size, 
            num_heads = 1, 
            feat_drop= gat_conv_feat_drop, 
            attn_drop= gat_conv_attn_drop, 
            negative_slope=0.2, 
            residual=False, 
            activation=None, 
            allow_zero_in_degree=False
        )

        self.activation = nn.ReLU() if activation == 'relu' else nn.Tanh()
    
    def forward(self, graph, feats):
        feats = self.attentions(graph, feats)
        feats = feats.reshape(graph.num_nodes(), -1)
        feats = self.out_att(graph, feats)
        return feats

if __name__ == '__main__':
    model = GATConvOp()
    
    import torch, dgl
    h = torch.rand(10, 128)
    G = dgl.graph(data = ([1,2,3,4,5], [6,7,8,9,0]), num_nodes=10)
    G = dgl.add_self_loop(G)
    output = model(G, h)
    print(output.min(), output.max())
