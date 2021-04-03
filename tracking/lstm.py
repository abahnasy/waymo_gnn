import torch
import torch.nn as nn



class LSTMfeat(nn.Module):
    def __init__(self,input_dim, hidden_dim, n_layers, batch_first=True):
        super(LSTMfeat, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm_layer = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)

    def forward(self, x):
        batch_size = x.shape[0]
        hidden_state = torch.randn(self.n_layers, batch_size, self.hidden_dim).cuda()
        cell_state = torch.randn(self.n_layers, batch_size, self.hidden_dim).cuda()
        hidden = (hidden_state, cell_state)
        out, hidden = self.lstm_layer(x, hidden)
        # print(out)
        # return last output, many-to-one
        out = out.transpose(0,1)
        out = out[-1,...]
        return out

if __name__ == "__main__":


    input_dim = 5
    hidden_dim = 10
    n_layers = 1
    batch_size = 3
    seq_len = 5

    inp = torch.randn(batch_size, seq_len, input_dim)

    lstm = LSTMfeat(input_dim, hidden_dim, n_layers)
    out = lstm(inp)
    print("Output shape: ", out.shape)
    print(out)



    