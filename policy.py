import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Policy, self).__init__()
        self.action_dim = action_dim
        self.out = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)
        )

    def forward(self, state):
        out = self.out(state)
        mu, log_var = out[:, :self.action_dim], out[:, self.action_dim:]
        return mu, torch.clamp(log_var, -5, 2)
