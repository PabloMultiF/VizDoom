import torch
from torch import nn

class ActorCriticNetwork(nn.Module):
    def __init__(self, in_channels, n_output):
        super(ActorCriticNetwork, self).__init__()

        network = [
            torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2304,512),
            nn.ReLU(), 
        ]
        
        self.network = nn.Sequential(*network)
        self.actor_output = nn.Sequential(
            nn.Linear(512, n_output), 
            nn.Softmax(dim=1)
        )
        self.critic_output = nn.Sequential(
            nn.Linear(512, 1), 
        )

    def forward(self, state):
        network_output = self.network(state)
        value = self.critic_output(network_output)
        log_probs = self.actor_output(network_output)
        return log_probs, value