import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv

from tracking.models.registry import CONV_OPERATORS

@CONV_OPERATORS.register_module
class SAGEConvOp(nn.Module):
    """ https://arxiv.org/pdf/1706.02216.pdf
    """
    
    def __init__(self, feature_size=128, num_gnn_layers=4, activation='relu', **kwargs):
        super(SAGEConvOp, self).__init__()
        self.feature_size = feature_size
        self.num_gnn_layers = num_gnn_layers
        
        
        self.graph_conv_layers = nn.ModuleList([
            SAGEConv(
                self.feature_size, 
                self.feature_size, 
                norm=None, 
                bias=True, 
                aggregator_type= 'pool', 
                feat_drop = 0.0
            ) for _ in range(self.num_gnn_layers)
        ])
        self.activation = nn.ReLU() if activation == 'relu' else nn.Tanh()
    
    def forward(self, graph, feats):
        for layer in self.graph_conv_layers:
            feats = layer(graph, feats)
            feats = self.activation(feats)
        return feats

if __name__ == '__main__':
    model = SAGEConvOp()
    
    import torch, dgl
    h = torch.rand(10, 128)
    G = dgl.graph(data = ([1,2,3,4,5], [6,7,8,9,0]), num_nodes=10)
    G = dgl.add_self_loop(G)
    output = model(G, h)
    print(output.min(), output.max())
