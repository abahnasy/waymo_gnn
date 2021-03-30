""" 2-Layer MLP to extract motion feature for detected objects
"""

import torch
import torch.nn as nn

class TwoLayersMLP (nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayersMLP, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output


class EdgeRegressionMLP(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(EdgeRegressionMLP, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()
    
    
    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output

if __name__ == "__main__":
    mlp = TwoLayersMLP(7,10,100)
    input = torch.rand(1,7)
    output = mlp(input)
    print(output.shape)