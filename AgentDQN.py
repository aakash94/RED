import torch.nn as nn
import torch.nn.functional as F

class AgentDQN(nn.Module):

    def __init__(self, n_ip, n_op):
        super(AgentDQN, self).__init__()
        self.fc1 = nn.Linear(n_ip, n_ip *16)
        self.fc2 = nn.Linear(n_ip *16, n_ip *16)
        self.fc3 = nn.Linear(n_ip *16, n_op)

    def forward(self ,x):
        x=F.relu(self.fc1(x))
        x=F.relu( self.fc2(x))
        x=self.fc3(x)
        return x

