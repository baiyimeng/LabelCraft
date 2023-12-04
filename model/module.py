import torch.nn as nn
import torch

class Dice(nn.Module):
    """The Data Adaptive Activation Function in DIN,which can be viewed as a generalization of PReLu and can adaptively adjust the rectified point according to distribution of input data.
    Input shape:
        - 2 dims: [batch_size, embedding_size(features)]
        - 3 dims: [batch_size, num_features, embedding_size(features)]
    Output shape:
        - Same shape as input.
    References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
        - https://github.com/zhougr1993/DeepInterestNetwork, https://github.com/fanoping/DIN-pytorch
    """

    def __init__(self, emb_size, dim=2, epsilon=1e-8):
        super(Dice, self).__init__()
        assert dim == 2 or dim == 3

        self.bn = nn.BatchNorm1d(emb_size, eps=epsilon)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim

        # wrap alpha in nn.Parameter to make it trainable
        if self.dim == 2:
            self.alpha = nn.Parameter(torch.zeros((emb_size,)))
        else:
            self.alpha = nn.Parameter(torch.zeros((emb_size, 1)))

    def forward(self, x):
        assert x.dim() == self.dim
        if self.dim == 2:
            x_p = self.sigmoid(self.bn(x))  # B * embed
            out = self.alpha * (1 - x_p) * x + x_p * x 
        else:
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x 
            out = torch.transpose(out, 1, 2)
        return out

class FullyConnectedLayer(nn.Module):

    def __init__(self, input_size, hidden_unit, batch_norm=False, activation='relu', sigmoid=False, dropout=None, dice_dim=None):
        super(FullyConnectedLayer, self).__init__()
        assert len(hidden_unit) >= 1 
        self.sigmoid = sigmoid

        layers = []
        layers.append(nn.Linear(input_size, hidden_unit[0]))  # 加入第一层
        
        for i in range(len(hidden_unit)-1):
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_unit[i]))
            
            if activation.lower() == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation.lower() == 'tanh':
                layers.append(nn.Tanh())
            elif activation.lower() == 'leakyrelu':
                layers.append(nn.LeakyReLU())
            elif activation.lower() == 'dice':
                assert dice_dim
                layers.append(Dice(hidden_unit[i], dim=dice_dim))
            else:
                raise NotImplementedError

            if dropout is not None:
                layers.append(nn.Dropout(dropout))
            
            layers.append(nn.Linear(hidden_unit[i], hidden_unit[i+1]))
        
        self.fc = nn.Sequential(*layers)
        if self.sigmoid:
            self.output_layer = nn.Sigmoid()
        

    def forward(self, x):
        return self.output_layer(self.fc(x)) if self.sigmoid else self.fc(x) 
        


