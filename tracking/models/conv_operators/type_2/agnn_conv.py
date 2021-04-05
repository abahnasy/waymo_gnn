import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import AGNNConv

from tracking.models.registry import CONV_OPERATORS

@CONV_OPERATORS.register_module
class AGNNConvOp(nn.Module):
    """ https://arxiv.org/abs/1803.03735
    """
    
    def __init__(self, agnn_conv_init_beta=1.0, agnn_conv_learn_beta=True, num_gnn_layers=4, activation='relu', **kwargs):
        super(AGNNConvOp, self).__init__()
        self.agnn_conv_init_beta = agnn_conv_init_beta
        self.agnn_conv_learn_beta = agnn_conv_learn_beta
        self.num_gnn_layers = num_gnn_layers
        
        
        self.graph_conv_layers = nn.ModuleList([
            AGNNConv(agnn_conv_init_beta, agnn_conv_learn_beta) for _ in range(self.num_gnn_layers)
        ])
        self.activation = nn.ReLU() if activation == 'relu' else nn.Tanh()
    
    def forward(self, graph, feats):
        for layer in self.graph_conv_layers:
            feats = layer(graph, feats)
            feats = self.activation(feats)
        return feats

if __name__ == '__main__':
    model = AGNNConvOp()
    
    import torch, dgl
    h = torch.rand(10, 128)
    
    
    G = dgl.graph(data = ([1,2,3,4,5], [6,7,8,9,0]), num_nodes=10)
    G = dgl.add_self_loop(G)

    
    output = model(G, h)
    print(output.min(), output.max())
