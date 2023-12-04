import torch
import torch.nn as nn


class Labeler(nn.Module):
    def __init__(self, feedback_num, dim_num):
        super(Labeler, self).__init__()
        self.feedback_num = feedback_num
        self.fc1 = nn.Linear(feedback_num, dim_num, bias=False)
        self.fc2 = nn.Linear(dim_num, dim_num, bias=False)
        self.fc3 = nn.Linear(dim_num, 1, bias=False)
        self.acti = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = x[:, :self.feedback_num]
        assert x.shape[-1] == self.feedback_num
        x = x.to(torch.float32)
        x = self.fc1(x)
        x = self.acti(x)
        x = self.fc2(x)
        x = self.acti(x)
        x = self.fc3(x)
        return self.sigmoid(x)
