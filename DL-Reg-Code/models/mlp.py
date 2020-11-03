'''4-layers MLP_RELU in PyTorch.

Accordign to Table 9 in Paper: "Dropout: A Simple Way to Prevent Neural Networks from
Overfitting".
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class _MLP(nn.Module):
    def __init__(self, in_ch, num_classes):
        super(_MLP, self).__init__()
        self.linear_1 = nn.Linear(in_ch, 1024)
        self.linear_2 = nn.Linear(1024, 1024)
        self.linear_3 = nn.Linear(1024, 2048)
        self.linear_4 = nn.Linear(2048, num_classes)
        self.drop_layer = nn.Dropout(p=0.5)
        self.drop_input_layer = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.linear_1(x.view(x.size()[0], -1))
        out = self.drop_input_layer(out)
        out = self.linear_2(F.relu(out))
        out = self.drop_layer(out)
        out = self.linear_3(F.relu(out))
        out = self.drop_layer(out)
        out = self.linear_4(F.relu(out))
        out = F.relu(out)
        return out


def mlp_relu(in_ch, num_classes):

    return _MLP(in_ch=in_ch, num_classes=num_classes)
    
    


# test()
