import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import EdgeConv

from tracking.models.registry import CONV_OPERATORS

@CONV_OPERATORS.register_module
class EdgeConvOp(nn.Module):
    """ https://arxiv.org/pdf/1801.07829
    """
    
    def __init__(self, feature_size=128, num_gnn_layers=4, activation='relu', **kwargs):
        super(EdgeConvOp, self).__init__()
        self.feature_size = feature_size
        self.num_gnn_layers = num_gnn_layers
        
        
        self.graph_conv_layers = nn.ModuleList([
            EdgeConv(self.feature_size, self.feature_size) for _ in range(self.num_gnn_layers)
        ])
        self.activation = nn.ReLU() if activation == 'relu' else nn.Tanh()
    
    def forward(self, graph, feats):
        for layer in self.graph_conv_layers:
            feats = layer(graph, feats)
            feats = self.activation(feats)
        return feats

if __name__ == '__main__':
    model = EdgeConvOp()
    
    import torch, dgl
    h = torch.rand(10, 128)
    G = dgl.graph(data = ([1,2,3,4,5], [6,7,8,9,0]), num_nodes=10)
    G = dgl.add_self_loop(G)
    output = model(G, h)
    print(output.min(), output.max())


