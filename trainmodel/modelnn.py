#import torch.nn as nn
from torch import nn
import torch
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.flatten = torch.flatten()
        self.fc1 = nn.Linear(12288, 84)
        self.fc2 = nn.Linear(84,50)
        self.fc3 = nn.Linear(50,2)
    
    def forward(self,x):
        x = x.view(-1, 12288)
        #x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x        

model = Net()

if torch.cuda.is_available():
    model = model.cuda()
    
    